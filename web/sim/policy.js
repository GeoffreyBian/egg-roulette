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

function layernormGelu(x, g, beta) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) {
    const z = (x[i] - mean) * inv * g[i] + beta[i];
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
