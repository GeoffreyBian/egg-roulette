// Browser port of eggsim/camera.py: one ray per pixel through MuJoCo's mj_multiRay against the collision
// geometry, flat colour per object, one directional light. Settings and colours come from assets/camera.json,
// written from camera.py, so both sides read the same numbers.

export class RayCamera {
  constructor(mj, m, spec, name) {
    this.mj = mj; this.m = m; this.spec = spec;
    this.cam = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA.value, name);
    if (this.cam < 0) throw new Error(`no camera ${name}`);
    const body = m.cam_bodyid[this.cam];
    this.exclude = body > 0 ? body : -1;               // the camera's own lens
    const res = spec.res, t = Math.tan((m.cam_fovy[this.cam] * Math.PI / 180) / 2);
    this.res = res;
    this.local = new Float64Array(3 * res * res);
    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const x = ((j + 0.5) / res * 2 - 1) * t, y = -(((i + 0.5) / res * 2 - 1) * t), z = -1;
        const n = Math.sqrt(x * x + y * y + z * z), k = 3 * (i * res + j);
        this.local[k] = x / n; this.local[k + 1] = y / n; this.local[k + 2] = z / n;
      }
    }
    const nr = res * res;
    this.vec = new Array(3 * nr).fill(0);
    this.gid = new mj.IntBuffer(nr);
    this.dist = new mj.DoubleBuffer(nr);
    this.normal = new mj.DoubleBuffer(3 * nr);
    this.image = new Uint8Array(3 * nr);
  }

  // (res * res * 3) uint8, row-major from the top-left. Needs cam and geom poses current.
  render(d) {
    const { mj, m, spec } = this, nr = this.res * this.res;
    const R = d.cam_xmat.subarray(9 * this.cam, 9 * this.cam + 9), L = this.local, v = this.vec;
    for (let r = 0; r < nr; r++) {
      const a = L[3 * r], b = L[3 * r + 1], c = L[3 * r + 2];
      v[3 * r] = a * R[0] + b * R[1] + c * R[2];
      v[3 * r + 1] = a * R[3] + b * R[4] + c * R[5];
      v[3 * r + 2] = a * R[6] + b * R[7] + c * R[8];
    }
    const p = d.cam_xpos.subarray(3 * this.cam, 3 * this.cam + 3);
    mj.mj_multiRay(m, d, [p[0], p[1], p[2]], v, spec.groups, true, this.exclude, this.gid, this.dist, this.normal, nr, spec.cutoff);
    const gid = this.gid.GetView(), nrm = this.normal.GetView(), img = this.image;
    const [lx, ly, lz] = spec.light, col = spec.colours, bg = spec.background;
    for (let r = 0; r < nr; r++) {
      let rgb0 = bg[0], rgb1 = bg[1], rgb2 = bg[2];
      const g = gid[r];
      if (g >= 0) {
        let nx = nrm[3 * r], ny = nrm[3 * r + 1], nz = nrm[3 * r + 2];
        if (nx * v[3 * r] + ny * v[3 * r + 1] + nz * v[3 * r + 2] > 0) { nx = -nx; ny = -ny; nz = -nz; }   // face the camera
        const lam = Math.min(1, Math.max(0, nx * lx + ny * ly + nz * lz));
        const s = spec.ambient + spec.diffuse * lam;
        rgb0 = col[3 * g] * s; rgb1 = col[3 * g + 1] * s; rgb2 = col[3 * g + 2] * s;
      }
      img[3 * r] = Math.floor(Math.min(1, Math.max(0, rgb0)) * 255 + 0.5);
      img[3 * r + 1] = Math.floor(Math.min(1, Math.max(0, rgb1)) * 255 + 0.5);
      img[3 * r + 2] = Math.floor(Math.min(1, Math.max(0, rgb2)) * 255 + 0.5);
    }
    return img;
  }
}

export class Cameras {
  constructor(mj, m, spec) { this.mj = mj; this.m = m; this.cams = spec.cameras.map((n) => new RayCamera(mj, m, spec, n)); }

  // mj_multiRay culls with bounding boxes from collision detection: run the position stage for the current qpos
  render(d) {
    this.mj.mj_fwdPosition(this.m, d);
    return this.cams.map((c) => c.render(d).slice());
  }
}
