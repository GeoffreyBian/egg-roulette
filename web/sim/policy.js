// Browser port of policy.Runner: the ACT-style MLP (float16 weights from eggsim/web_export.py), history
// stacking, action chunks and temporal ensembling.

import { KP_G, LEVER, Rng, gripToQ } from './env.js';

const TOUCH_DIMS = 3;

function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

// erf to ~1e-7 (Abramowitz and Stegun 7.1.26 is too coarse for an exact-GELU match; this is W. J. Cody's form)
function erf(x) {
  const t = 1 / (1 + 0.5 * Math.abs(x));
  const y = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? y : -y;
}

export class Policy {
  constructor(meta, buffer) {
    this.meta = meta;
    const w = halfToFloat(new Uint16Array(buffer));
    let off = 0;
    this.params = meta.shapes.map((shape) => {
      const n = shape.reduce((a, b) => a * b, 1), p = w.subarray(off, off + n);
      off += n;
      return { shape, p };
    });
    this.hidden = [new Float32Array(512), new Float32Array(512)];
  }

  static async load(meta, url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`policy ${url}: ${res.status}`);
    return new Policy(meta, await res.arrayBuffer());
  }

  forward(x) {
    const P = this.params;
    let h = Float32Array.from(x);
    // three blocks of Linear -> LayerNorm -> GELU, then the action head
    for (let L = 0; L < 3; L++) {
      const [W, b, g, beta] = P.slice(4 * L, 4 * L + 4);
      h = layernormGelu(linear(W, b, h), g.p, beta.p);
    }
    const [W, b] = P.slice(12, 14);
    return linear(W, b, h);
  }
}

function linear(W, b, x) {
  const [nOut, nIn] = W.shape, w = W.p, y = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = b.p[o];
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += w[row + i] * x[i];
    y[o] = s;
  }
  return y;
}

function layernormGelu(x, g, beta, rec = null) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  if (rec) { rec.inv = inv; rec.xhat = new Float32Array(n); rec.z = new Float32Array(n); }
  for (let i = 0; i < n; i++) {
    const xh = (x[i] - mean) * inv, z = xh * g[i] + beta[i];
    if (rec) { rec.xhat[i] = xh; rec.z[i] = z; }
    x[i] = 0.5 * z * (1 + erf(z / Math.SQRT2));
  }
  return x;
}

// Squeeze a servo could report from its own position error, written into the first touch channel; no load,
// no slip (policy.servo_estimate). o[3] is the jaw angle, o[16] the previous grip command.
function servoEstimate(o, rng) {
  const g = o[16], qCmd = gripToQ(g);
  let f = Math.max(KP_G * (o[3] - qCmd), 0) / LEVER * (g >= 0 ? 1 : 0);
  f = Math.max(f + rng.normal(0, 0.3), 0);                 // current sensing noise ~0.3 N
  const out = o.slice();
  out[o.length - 3] = f / 3.0; out[o.length - 2] = 0; out[o.length - 1] = 0;
  return out;
}

export class Runner {
  // touchSource: 'pads' (pad sensor, as the glove-trained policies expect) | 'servo' (squeeze estimated from
  // servo current) | 'off' (sensor unplugged: zeros)
  constructor(policy, { touchSource = 'pads', seed = 12345 } = {}) {
    this.pol = policy; this.touchSource = touchSource; this.seed = seed; this.reset();
  }

  reset() { this.hist = []; this.pending = []; this.lastChunk = null; this.rng = new Rng(this.seed); }

  act(oFull) {
    const meta = this.pol.meta;
    if (meta.proxy || this.touchSource === 'servo') oFull = servoEstimate(oFull, this.rng);
    const nObs = oFull.length;
    oFull = oFull.map((v, i) => (i >= nObs - TOUCH_DIMS
      ? (this.touchSource === 'off' ? 0 : v * meta.touch_mask[i - nObs + TOUCH_DIMS]) : v));
    const o = meta.touch ? oFull : oFull.slice(0, oFull.length - TOUCH_DIMS);
    this.hist.unshift(o);
    this.hist = this.hist.slice(0, meta.hist);
    while (this.hist.length < meta.hist) this.hist.push(this.hist[this.hist.length - 1]);
    const x = this.hist.flat().map((v, i) => (v - meta.mx[i]) / meta.sx[i]);
    const y = this.pol.forward(x);
    const K = meta.chunk, chunk = [];
    for (let k = 0; k < K; k++) chunk.push([0, 1, 2, 3].map((j) => y[4 * k + j] * meta.sy[4 * k + j] + meta.my[4 * k + j]));
    this.lastChunk = chunk;
    // temporal ensembling: average this step's predictions from every chunk still in flight
    this.pending = this.pending.filter(([, i]) => i + 1 < K).map(([c, i]) => [c, i + 1]);
    this.pending.push([chunk, 0]);
    const n = this.pending.length, a = [0, 0, 0, 0];
    let wsum = 0;
    this.pending.forEach(([c, i], j) => {
      const w = Math.exp(-0.25 * (n - 1 - j));
      wsum += w;
      for (let d = 0; d < 4; d++) a[d] += w * c[i][d];
    });
    return a.map((v) => v / wsum);
  }
}

// ---------------------------------------------------------------- pixel policies (eggsim/pixels.py)
// Per camera: conv 5x5/2 -> ReLU -> conv 3x3/2 -> ReLU -> conv 3x3/1 -> spatial softmax (32 keypoints).
// The softmax maps are kept: they are where the policy looks, drawn as the saliency overlay.
function conv2d(x, C, H, W, w, b, Cout, k, stride, pad, relu) {
  const Ho = Math.floor((H + 2 * pad - k) / stride) + 1, Wo = Math.floor((W + 2 * pad - k) / stride) + 1, n = Ho * Wo;
  // im2col: one contiguous row of n input values per (channel, tap), zero padding written in once
  const R = C * k * k, cols = new Float32Array(R * n);
  for (let c = 0; c < C; c++) {
    for (let di = 0; di < k; di++) {
      for (let dj = 0; dj < k; dj++) {
        const row = ((c * k + di) * k + dj) * n;
        for (let i = 0; i < Ho; i++) {
          const yi = i * stride - pad + di;
          if (yi < 0 || yi >= H) continue;
          const src = (c * H + yi) * W, dst = row + i * Wo;
          for (let j = 0; j < Wo; j++) {
            const xj = j * stride - pad + dj;
            if (xj >= 0 && xj < W) cols[dst + j] = x[src + xj];
          }
        }
      }
    }
  }
  // y = w (Cout x R) times cols (R x n), four output channels per pass so each input value is loaded once for four
  const y = new Float32Array(Cout * n);
  for (let o = 0; o < Cout; o += 4) {
    const m = Math.min(4, Cout - o);
    const y0 = o * n, y1 = y0 + n, y2 = y1 + n, y3 = y2 + n;
    for (let q = 0; q < m; q++) y.fill(b[o + q], y0 + q * n, y0 + (q + 1) * n);
    for (let r = 0; r < R; r++) {
      const w0 = w[o * R + r], w1 = m > 1 ? w[(o + 1) * R + r] : 0, w2 = m > 2 ? w[(o + 2) * R + r] : 0, w3 = m > 3 ? w[(o + 3) * R + r] : 0;
      const cr = r * n;
      if (m === 4) {
        for (let j = 0; j < n; j++) {
          const v = cols[cr + j];
          y[y0 + j] += w0 * v; y[y1 + j] += w1 * v; y[y2 + j] += w2 * v; y[y3 + j] += w3 * v;
        }
      } else {
        for (let q = 0; q < m; q++) { const wq = w[(o + q) * R + r], yq = y0 + q * n; for (let j = 0; j < n; j++) y[yq + j] += wq * cols[cr + j]; }
      }
    }
  }
  if (relu) for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
  return { y, H: Ho, W: Wo };
}

export class PixelPolicy {
  constructor(meta, buffer) {
    this.meta = meta;
    const w = halfToFloat(new Uint16Array(buffer));
    let off = 0;
    const take = (shape) => {
      const n = shape.reduce((a, b) => a * b, 1), p = w.subarray(off, off + n);
      off += n;
      return { shape, p };
    };
    this.params = meta.shapes.map(take);
  }

  static async load(meta, url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`policy ${url}: ${res.status}`);
    return new PixelPolicy(meta, await res.arrayBuffer());
  }

  // img: uint8 (64 * 64 * 3, HWC) -> keypoints [x(32), y(32)]; activations are kept for saliency()
  encode(img, e) {
    const P = this.params, R = this.meta.res, base = 7 * e;
    const x = new Float32Array(3 * R * R);
    for (let c = 0; c < 3; c++) for (let i = 0; i < R * R; i++) x[c * R * R + i] = img[3 * i + c] / 255 - 0.5;
    const h1 = conv2d(x, 3, R, R, P[base].p, P[base + 1].p, 16, 5, 2, 2, true);
    const h2 = conv2d(h1.y, 16, h1.H, h1.W, P[base + 2].p, P[base + 3].p, 32, 3, 2, 1, true);
    const h3 = conv2d(h2.y, 32, h2.H, h2.W, P[base + 4].p, P[base + 5].p, this.meta.keypoints, 3, 1, 1, false);
    const temp = Math.exp(P[base + 6].p[0]), K = this.meta.keypoints, n = h3.H * h3.W;
    const kx = new Float32Array(K), ky = new Float32Array(K), maps = new Float32Array(K * n);
    for (let c = 0; c < K; c++) {
      let mx = -Infinity;
      for (let i = 0; i < n; i++) mx = Math.max(mx, h3.y[c * n + i] / temp);
      let s = 0;
      for (let i = 0; i < n; i++) { const v = Math.exp(h3.y[c * n + i] / temp - mx); maps[c * n + i] = v; s += v; }
      for (let i = 0; i < n; i++) {
        const p = maps[c * n + i] / s, r = Math.floor(i / h3.W), q = i % h3.W;
        maps[c * n + i] = p;
        kx[c] += p * (-1 + 2 * q / (h3.W - 1)); ky[c] += p * (-1 + 2 * r / (h3.H - 1));
      }
    }
    return { kp: [...kx, ...ky], kx, ky, maps, size: h3.W, temp, h1, h2 };
  }

  // imgs: [workspace, wrist] uint8 images; prop: normalised proprio history
  forward(imgs, prop) {
    const P = this.params;
    this.enc = imgs.map((im, e) => this.encode(im, e));
    let h = Float32Array.from([...this.enc[0].kp, ...this.enc[1].kp, ...prop]);
    this.layers = [];
    for (let L = 0; L < 3; L++) {
      const [W, b, g, beta] = P.slice(14 + 4 * L, 18 + 4 * L);
      const u = linear(W, b, h);
      const rec = { u: Float32Array.from(u) };
      h = layernormGelu(u, g.p, beta.p, rec);
      this.layers.push(rec);
    }
    const [W, b] = P.slice(26, 28);
    return linear(W, b, h);
  }

  // Gradient saliency for the last forward(): sum over the first chunk action's 4 outputs (normalised units) of
  // |d output / d pixel|, summed over colour channels. Returns one Float32Array (res * res) per camera.
  saliency() {
    const P = this.params, R = this.meta.res, K = this.meta.keypoints;
    const out = [new Float32Array(R * R), new Float32Array(R * R)];
    const [Wa] = P.slice(26, 28);
    for (let j = 0; j < 4; j++) {
      // action head and trunk, backwards
      let g = new Float32Array(512);
      for (let i = 0; i < 512; i++) g[i] = Wa.p[j * 512 + i];
      for (let L = 2; L >= 0; L--) {
        const [W, , gam] = P.slice(14 + 4 * L, 18 + 4 * L), rec = this.layers[L];
        const n = 512, gx = new Float32Array(n);
        let m1 = 0, m2 = 0;
        for (let i = 0; i < n; i++) {
          const z = rec.z[i], cdf = 0.5 * (1 + erf(z / Math.SQRT2)), pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
          gx[i] = g[i] * (cdf + z * pdf) * gam.p[i];
          m1 += gx[i]; m2 += gx[i] * rec.xhat[i];
        }
        m1 /= n; m2 /= n;
        const gu = new Float32Array(n);
        for (let i = 0; i < n; i++) gu[i] = rec.inv * (gx[i] - m1 - rec.xhat[i] * m2);
        const [nOut, nIn] = W.shape, gin = new Float32Array(nIn);
        for (let o = 0; o < nOut; o++) { const go = gu[o], row = o * nIn; for (let i = 0; i < nIn; i++) gin[i] += W.p[row + i] * go; }
        g = gin;
      }
      // keypoints -> spatial softmax -> convs -> pixels, per camera
      for (let e = 0; e < 2; e++) {
        const enc = this.enc[e], n = enc.size * enc.size, base = 7 * e;
        const gf = new Float32Array(K * n);
        for (let k = 0; k < K; k++) {
          const gxk = g[e * 2 * K + k], gyk = g[e * 2 * K + K + k];
          for (let i = 0; i < n; i++) {
            const r = Math.floor(i / enc.size), q = i % enc.size;
            const X = -1 + 2 * q / (enc.size - 1), Y = -1 + 2 * r / (enc.size - 1);
            gf[k * n + i] = enc.maps[k * n + i] / enc.temp * (gxk * (X - enc.kx[k]) + gyk * (Y - enc.ky[k]));
          }
        }
        let gh2 = convBackInput(gf, K, enc.size, enc.size, P[base + 4].p, 32, enc.h2.H, enc.h2.W, 3, 1, 1);
        for (let i = 0; i < gh2.length; i++) if (enc.h2.y[i] <= 0) gh2[i] = 0;
        let gh1 = convBackInput(gh2, 32, enc.h2.H, enc.h2.W, P[base + 2].p, 16, enc.h1.H, enc.h1.W, 3, 2, 1);
        for (let i = 0; i < gh1.length; i++) if (enc.h1.y[i] <= 0) gh1[i] = 0;
        const gimg = convBackInput(gh1, 16, enc.h1.H, enc.h1.W, P[base].p, 3, R, R, 5, 2, 2);
        for (let c = 0; c < 3; c++) for (let i = 0; i < R * R; i++) out[e][i] += Math.abs(gimg[c * R * R + i]);
      }
    }
    return out;
  }
}

// gradient of a convolution with respect to its input (the transposed convolution)
function convBackInput(gy, Cout, Ho, Wo, w, C, H, W, k, stride, pad) {
  const gx = new Float32Array(C * H * W);
  for (let o = 0; o < Cout; o++) {
    for (let c = 0; c < C; c++) {
      for (let di = 0; di < k; di++) {
        for (let dj = 0; dj < k; dj++) {
          const wv = w[((o * C + c) * k + di) * k + dj];
          for (let i = 0; i < Ho; i++) {
            const yi = i * stride - pad + di;
            if (yi < 0 || yi >= H) continue;
            const gr = (o * Ho + i) * Wo, xr = (c * H + yi) * W;
            for (let j = 0; j < Wo; j++) {
              const xj = j * stride - pad + dj;
              if (xj >= 0 && xj < W) gx[xr + xj] += wv * gy[gr + j];
            }
          }
        }
      }
    }
  }
  return gx;
}

const PROPRIO = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16], TOUCH = [17, 18, 19];

export class PixelRunner {
  // cameras: a Cameras instance on the env's model; touchSource as in Runner
  constructor(policy, cameras, { touchSource = 'pads', seed = 12345 } = {}) {
    this.pol = policy; this.cameras = cameras; this.touchSource = touchSource; this.seed = seed; this.reset();
  }

  reset() { this.hist = []; this.pending = []; this.rng = new Rng(this.seed); }

  // oFull: flat observation (env.obs()); d: MjData to render the cameras from. frames may be given (parity test).
  act(oFull, d, frames = null) {
    const meta = this.pol.meta;
    if (this.touchSource === 'servo') oFull = servoEstimate(oFull, this.rng);
    if (this.touchSource === 'off') oFull = oFull.map((v, i) => (i >= 17 ? 0 : v));
    const prop = (meta.touch ? [...PROPRIO, ...TOUCH] : PROPRIO).map((i) => oFull[i]);
    this.hist.unshift(prop);
    this.hist = this.hist.slice(0, meta.hist);
    while (this.hist.length < meta.hist) this.hist.push(this.hist[this.hist.length - 1]);
    const x = this.hist.flat().map((v, i) => (v - meta.mp[i]) / meta.sp[i]);
    this.frames = frames ?? this.cameras.render(d);
    const y = this.pol.forward(this.frames, x);
    const K = meta.chunk, chunk = [];
    for (let k = 0; k < K; k++) chunk.push([0, 1, 2, 3].map((j) => y[4 * k + j] * meta.sy[4 * k + j] + meta.my[4 * k + j]));
    this.pending = this.pending.filter(([, i]) => i + 1 < K).map(([c, i]) => [c, i + 1]);
    this.pending.push([chunk, 0]);
    const n = this.pending.length, a = [0, 0, 0, 0];
    let wsum = 0;
    this.pending.forEach(([c, i], j) => {
      const w = Math.exp(-0.25 * (n - 1 - j));
      wsum += w;
      for (let q = 0; q < 4; q++) a[q] += w * c[i][q];
    });
    return a.map((v) => v / wsum);
  }
}
