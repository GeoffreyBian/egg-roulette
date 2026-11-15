// Browser port of eggsim/env.py: same constants, same control loop, same contact sensing.
// The physics is MuJoCo itself (WebAssembly build, same version as the Python package), running the MJCF
// that eggsim/web_export.py writes. Only the glue around mj_step lives here.

export const SIM_DT = 0.002, CTRL_HZ = 30;
export const SUBSTEPS = Math.round(1 / CTRL_HZ / SIM_DT);
export const EGG_R = 0.021, EGG_H = 0.026, EGG_R_SD = 0.0005;
export const MU = 0.8, PAD_TORSION = 0.012;
const CARTON_ORIGIN = [0.155, 0.04], SLOT_PITCH = 0.064;
export const SLOTS = [];
for (let j = 0; j < 3; j++) for (let i = 0; i < 2; i++)
  SLOTS.push([CARTON_ORIGIN[0] + SLOT_PITCH * i, CARTON_ORIGIN[1] + SLOT_PITCH * j]);
export const CARTON_FLOOR = 0.006;
export const BASKET_C = [0.17, -0.13], BASKET_HALF = 0.075;
const PARK = [5.0, 5.0, 0.03];
const DEBRIS = ['yolk', 'white', 'shellA', 'shellB'];

// egg kinds: [mass kg, crack N]. Kinds past the first two never appear in training.
export const KINDS = {
  raw: { mass: 0.060, crack: 2.2 },
  stone: { mass: 0.300, crack: Infinity },
  raw_fragile: { mass: 0.060, crack: 1.6 },
  medium: { mass: 0.150, crack: 3.5 },
  stone_light: { mass: 0.180, crack: Infinity },
  stone_heavy: { mass: 0.400, crack: Infinity },
};

export const LEVER = 0.060, V_MAX = 0.22;
export const DROP_SPOTS = [];
for (const dy of [0.045, 0.0, -0.045]) for (const dx of [-0.025, 0.025])
  DROP_SPOTS.push([BASKET_C[0] + dx, BASKET_C[1] + dy]);
export const PICK_ORDER = [0, 1, 2, 3, 4, 5];
export const HOME = [0.18, -0.02, 0.095];
const WS_LO = [0.12, -0.24, 0.025], WS_HI = [0.34, 0.24, 0.105];
const READY = [0.0, -0.4, 0.3, 1.5];
export const Q_OPEN = 0.75, Q_MIN = -0.10;
const PINCH_LOCAL = [0.0117, -0.0002, -0.0830];
export const KP_G = 1.2, KD_G = 0.03, TAU_LIM = 0.35;
const W_ORI = 0.006;
const ROLL = 1.57 - Math.PI;

const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const gripToQ = (g) => Q_OPEN + (Q_MIN - Q_OPEN) * (clip(g, -1, 1) + 1) / 2;

// seeded RNG (mulberry32) with Box-Muller normals, so a seed always replays the same episode
export class Rng {
  constructor(seed) { this.s = seed >>> 0; this.spare = null; }
  uniform(lo = 0, hi = 1) {
    let t = (this.s += 0x6D2B79F5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return lo + (hi - lo) * (((t ^ (t >>> 14)) >>> 0) / 4294967296);
  }
  normal(mu = 0, sd = 1) {
    if (this.spare !== null) { const z = this.spare; this.spare = null; return mu + sd * z; }
    const u = Math.max(this.uniform(), 1e-12), v = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return mu + sd * r * Math.cos(2 * Math.PI * v);
  }
  int(n) { return Math.floor(this.uniform() * n); }
}

export class EggEnv {
  // opts: mu (pad friction), massScale, visionNoise, seed
  constructor(mj, model, opts = {}) {
    this.mj = mj; this.m = model;
    this.d = new mj.MjData(model);
    this._ikd = new mj.MjData(model);
    const m = model, id = (type, name) => {
      const i = mj.mj_name2id(m, mj.mjtObj[type].value, name);
      if (i < 0) throw new Error(`no ${type} ${name}`);
      return i;
    };
    this.id = id;
    this.nEggs = 6;
    this.rng = new Rng(opts.seed ?? 1);
    this.visionNoise = opts.visionNoise ?? 0.003;
    this.massScale = opts.massScale ?? 1.0;
    this.site = id('mjOBJ_SITE', 'gripperframe');
    const arm = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex'];
    const jnt = arm.map((n) => id('mjOBJ_JOINT', n));
    this.armDofs = jnt.map((j) => m.jnt_dofadr[j]);
    this.armQpos = [...arm, 'wrist_roll', 'gripper'].map((n) => m.jnt_qposadr[id('mjOBJ_JOINT', n)]);
    this.allArmDofs = [...arm, 'wrist_roll', 'gripper'].map((n) => m.jnt_dofadr[id('mjOBJ_JOINT', n)]);
    this.armRange = jnt.map((j) => [m.jnt_range[2 * j], m.jnt_range[2 * j + 1]]);
    this.gripAct = id('mjOBJ_ACTUATOR', 'gripper');
    this.gripDof = m.jnt_dofadr[id('mjOBJ_JOINT', 'gripper')];
    this.gripperBody = id('mjOBJ_BODY', 'gripper');
    this.movingJaw = id('mjOBJ_BODY', 'moving_jaw_so101_v1');
    this.jawGeoms = new Set();
    for (let g = 0; g < m.ngeom; g++)
      if (m.geom_bodyid[g] === this.gripperBody || m.geom_bodyid[g] === this.movingJaw) this.jawGeoms.add(g);
    this.eggBody = [], this.eggGeom = [], this.eggQadr = [], this.eggDadr = [], this.debris = [];
    for (let k = 0; k < this.nEggs; k++) {
      this.eggBody.push(id('mjOBJ_BODY', `egg${k}`));
      this.eggGeom.push(id('mjOBJ_GEOM', `egg${k}`));
      const j = id('mjOBJ_JOINT', `egg${k}`);
      this.eggQadr.push(m.jnt_qposadr[j]); this.eggDadr.push(m.jnt_dofadr[j]);
      this.debris.push(DEBRIS.map((p) => id('mjOBJ_BODY', `${p}${k}`)));
    }
    this.touchAdr = ['touch_fixed', 'touch_moving', 'gripper_torque'].map((n) => m.sensor_adr[id('mjOBJ_SENSOR', n)]);
    this.f6 = new mj.DoubleBuffer(6);
    this.v6a = new mj.DoubleBuffer(6);
    this.v6b = new mj.DoubleBuffer(6);
    this.jacp = new mj.DoubleBuffer(3 * m.nv);
    this.jacr = new mj.DoubleBuffer(3 * m.nv);
    this.setFriction(opts.mu ?? MU);
    this.eggs = []; this.eggR = [];
    this.target = 0; this.drop = [...BASKET_C]; this.eeTarget = [...HOME];
    this.prevAction = [0, 0, 0, 0]; this.t = 0;
    this.last = { grip: 0, shear: 0, slip: 0 };
  }

  setFriction(mu) {
    this.mu = mu;
    for (const g of this.jawGeoms) this.m.geom_friction.set([mu, PAD_TORSION, 0.002], 3 * g);
  }

  // per-egg physical parameters; kinds: names or {mass, crack} objects
  props(kind) { return typeof kind === 'string' ? KINDS[kind] : kind; }

  _geomsOf(b) {
    const m = this.m, out = [];
    for (let g = m.body_geomadr[b]; g < m.body_geomadr[b] + m.body_geomnum[b]; g++) out.push(g);
    return out;
  }

  // model edits shared by reset() and the parity test: egg sizes, masses, debris parked
  configure(kinds, eggR) {
    const m = this.m;
    for (let k = 0; k < this.nEggs; k++) {
      m.body_gravcomp[this.eggBody[k]] = 0;
      m.geom_contype[this.eggGeom[k]] = 1; m.geom_conaffinity[this.eggGeom[k]] = 1;
      for (const b of this.debris[k]) {
        m.body_gravcomp[b] = 1;
        for (const g of this._geomsOf(b)) { m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0; }
      }
    }
    this.eggs = kinds.map((kind, k) => ({ kind, slot: k % SLOTS.length, cracked: false, peak: 0 }));
    this.eggR = eggR;
    for (let k = 0; k < this.nEggs; k++) {
      const r = eggR[k], g = this.eggGeom[k];
      m.geom_size.set([r, r, r * EGG_H / EGG_R], 3 * g);
      m.geom_rbound[g] = r * EGG_H / EGG_R;
      const mass = this.props(kinds[k]).mass * this.massScale, b = this.eggBody[k];
      m.body_mass[b] = mass;
      const ixy = mass / 5 * (EGG_R ** 2 + EGG_H ** 2);
      m.body_inertia.set([ixy, ixy, mass / 5 * 2 * EGG_R ** 2], 3 * b);
    }
    this.mj.mj_setConst(m, this.d);
  }

  reset(kinds, { jitter = 0.002, present = null } = {}) {
    const m = this.m, d = this.d, rng = this.rng;
    kinds = kinds ?? Array.from({ length: this.nEggs }, () => (rng.uniform() < 0.5 ? 'raw' : 'stone'));
    const eggR = [];
    for (let k = 0; k < this.nEggs; k++) eggR.push(EGG_R + clip(rng.normal(0, EGG_R_SD), -2 * EGG_R_SD, 2 * EGG_R_SD));
    this.configure(kinds, eggR);
    this.mj.mj_resetData(m, d);
    for (let k = 0; k < this.nEggs; k++) {
      const s = k % SLOTS.length, q = this.eggQadr[k];
      const x = SLOTS[s][0] + rng.uniform(-jitter, jitter), y = SLOTS[s][1] + rng.uniform(-jitter, jitter);
      let p = [x, y, CARTON_FLOOR + eggR[k] * EGG_H / EGG_R + 0.0005];
      if (present && !present.includes(k)) p = [PARK[0] + k * 0.2, PARK[1], PARK[2]];
      d.qpos.set([...p, 1, 0, 0, 0], q);
    }
    READY.forEach((v, i) => { d.qpos[this.armQpos[i]] = v; });
    d.qpos[this.armQpos[4]] = ROLL;
    d.qpos[this.armQpos[5]] = Q_OPEN;
    this.eeTarget = [...HOME];
    const q = this._ik(HOME, 200);
    q.forEach((v, i) => { d.qpos[this.armQpos[i]] = v; d.ctrl[i] = v; });
    d.ctrl[4] = ROLL;
    d.ctrl[this.gripAct] = 0.2;
    this.mj.mj_forward(m, d);
    for (let i = 0; i < 150; i++) this.mj.mj_step(m, d);    // let the eggs settle on their supports
    this.prevAction = [0, 0, 0, 0]; this.t = 0; this.target = 0;
    this.last = { grip: 0, shear: 0, slip: 0 };
  }

  setTarget(k, drop = null) { this.target = k; this.drop = drop ? [...drop] : [...BASKET_C]; this.t = 0; }

  // the egg-roulette episode: target the i-th egg in pick order, later eggs still in the carton
  setupEpisode(kind, i = null, others = null) {
    const rng = this.rng;
    i = i ?? rng.int(6);
    const k = PICK_ORDER[i];
    const kinds = others ? [...others] : Array.from({ length: 6 }, () => (rng.uniform() < 0.5 ? 'raw' : 'stone'));
    kinds[k] = kind;
    this.reset(kinds, { present: PICK_ORDER.slice(i) });
    this.setTarget(k, DROP_SPOTS[i]);
    return k;
  }

  // ---------- kinematics ----------
  ee(dd = this.d) {
    const b = this.gripperBody, R = dd.xmat.subarray(9 * b, 9 * b + 9), p = dd.xpos.subarray(3 * b, 3 * b + 3);
    const [lx, ly, lz] = PINCH_LOCAL;
    return [p[0] + R[0] * lx + R[1] * ly + R[2] * lz, p[1] + R[3] * lx + R[4] * ly + R[5] * lz,
      p[2] + R[6] * lx + R[7] * ly + R[8] * lz];
  }

  _ik(pDes, iters = 40) {
    const mj = this.mj, m = this.m, dd = this._ikd;
    dd.qpos.set(this.d.qpos);
    const nv = m.nv, jp = this.jacp, jr = this.jacr;
    for (let it = 0; it < iters; it++) {
      mj.mj_kinematics(m, dd);
      mj.mj_comPos(m, dd);
      const p = this.ee(dd);
      const sx = dd.site_xmat.subarray(9 * this.site, 9 * this.site + 9);
      const a = [sx[0], sx[3], sx[6]];
      // e = [p_des - p, W * (a x down)], down = (0, 0, -1)
      const e = [pDes[0] - p[0], pDes[1] - p[1], pDes[2] - p[2], W_ORI * -a[1], W_ORI * a[0], 0];
      mj.mj_jac(m, dd, jp, jr, p, this.gripperBody);
      const JP = jp.GetView(), JR = jr.GetView();
      const J = [];                                        // 6 x 4
      for (let r = 0; r < 3; r++) J.push(this.armDofs.map((c) => JP[r * nv + c]));
      for (let r = 0; r < 3; r++) J.push(this.armDofs.map((c) => W_ORI * JR[r * nv + c]));
      const A = [];
      for (let i = 0; i < 6; i++) {
        A.push([]);
        for (let j = 0; j < 6; j++) {
          let s = 0;
          for (let c = 0; c < 4; c++) s += J[i][c] * J[j][c];
          A[i].push(s + (i === j ? 1e-4 : 0));
        }
      }
      const y = solve(A, e);
      for (let c = 0; c < 4; c++) {
        let dq = 0;
        for (let i = 0; i < 6; i++) dq += J[i][c] * y[i];
        dq = clip(dq, -0.15, 0.15);
        const adr = this.armQpos[c], [lo, hi] = this.armRange[c];
        dd.qpos[adr] = clip(dd.qpos[adr] + dq, lo, hi);
      }
      if (Math.hypot(e[0], e[1], e[2]) < 2e-4) break;
    }
    return this.armQpos.slice(0, 4).map((a) => dd.qpos[a]);
  }

  // ---------- sensing ----------
  eggPos(k = this.target) { return Array.from(this.d.xpos.subarray(3 * this.eggBody[k], 3 * this.eggBody[k] + 3)); }

  // squeeze (smaller of the two jaws' summed normal forces), load (upward force on the egg) and slip speed
  touch(k = this.target) {
    const mj = this.mj, m = this.m, d = this.d, gEgg = this.eggGeom[k];
    const normal = [0, 0], force = [0, 0, 0];
    let nC = 0;
    const con = d.contact;
    for (let i = 0; i < d.ncon; i++) {
      const c = con.get(i);
      let other, sign;
      if (c.geom1 === gEgg) { other = c.geom2; sign = -1; }
      else if (c.geom2 === gEgg) { other = c.geom1; sign = 1; }
      if (other === undefined || !this.jawGeoms.has(other)) { c.delete(); continue; }
      mj.mj_contactForce(m, d, i, this.f6);
      const f = this.f6.GetView(), fr = c.frame;
      normal[m.geom_bodyid[other] === this.movingJaw ? 1 : 0] += Math.abs(f[0]);
      for (let j = 0; j < 3; j++) force[j] += sign * (f[0] * fr[j] + f[1] * fr[3 + j] + f[2] * fr[6 + j]);
      nC++;
      c.delete();
    }
    con.delete();
    let slip = 0;
    if (nC) {
      const b = this.gripperBody, R = d.xmat.subarray(9 * b, 9 * b + 9);
      mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY.value, this.eggBody[k], this.v6a, 0);
      mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY.value, b, this.v6b, 0);
      const ve = this.v6a.GetView(), vg = this.v6b.GetView();
      const pe = d.xpos.subarray(3 * this.eggBody[k]), pg = d.xpos.subarray(3 * b);
      const r = [pe[0] - pg[0], pe[1] - pg[1], pe[2] - pg[2]];
      const vp = [vg[3] + vg[1] * r[2] - vg[2] * r[1], vg[4] + vg[2] * r[0] - vg[0] * r[2],
        vg[5] + vg[0] * r[1] - vg[1] * r[0]];
      const dv = [ve[3] - vp[0], ve[4] - vp[1], ve[5] - vp[2]];
      // R^T dv
      slip = Math.hypot(R[0] * dv[0] + R[3] * dv[1] + R[6] * dv[2], R[1] * dv[0] + R[4] * dv[1] + R[7] * dv[2],
        R[2] * dv[0] + R[5] * dv[1] + R[8] * dv[2]);
    }
    return { grip: Math.min(normal[0], normal[1]), shear: Math.max(0, force[2]), slip };
  }

  // every active contact on the given bodies (eggs and their debris), for drawing: {p, f (force on the body), jaw}
  contacts(bodies) {
    const mj = this.mj, m = this.m, d = this.d, out = [];
    const con = d.contact;
    for (let i = 0; i < d.ncon; i++) {
      const c = con.get(i);
      const on2 = bodies.has(m.geom_bodyid[c.geom2]), on1 = bodies.has(m.geom_bodyid[c.geom1]);
      if (on1 || on2) {
        mj.mj_contactForce(m, d, i, this.f6);
        const f = this.f6.GetView(), fr = c.frame, sign = on2 ? 1 : -1;   // contact force acts on geom2
        const w = [0, 1, 2].map((j) => sign * (f[0] * fr[j] + f[1] * fr[3 + j] + f[2] * fr[6 + j]));
        out.push({ p: Array.from(c.pos), f: w, jaw: this.jawGeoms.has(on2 ? c.geom1 : c.geom2) });
      }
      c.delete();
    }
    con.delete();
    return out;
  }

  nativeTouch() { const s = this.d.sensordata; return [s[this.touchAdr[0]], s[this.touchAdr[1]]]; }
  servoTorque() { return this.d.sensordata[this.touchAdr[2]]; }

  // flat observation, same layout as env.flat_obs (touch always included; the runner drops it if unused)
  obs() {
    const ee = this.ee(), eggTrue = this.eggPos(), n = this.visionNoise, rng = this.rng;
    const egg = eggTrue.map((v) => v + rng.normal(0, n));
    const basket = [this.drop[0], this.drop[1], 0];
    const ti = this.last;
    return [...ee, this.d.qpos[this.armQpos[5]], ...egg, egg[0] - ee[0], egg[1] - ee[1], egg[2] - ee[2],
      basket[0] - ee[0], basket[1] - ee[1], basket[2] - ee[2], ...this.prevAction,
      ti.grip / 3.0, ti.shear / 3.0, Math.min(ti.slip, 0.3) / 0.1];
  }

  // ---------- stepping ----------
  // step() = beginStep() + SUBSTEPS x substep() + endStep(). The viewer calls the pieces itself so that it can
  // draw between physics steps (smooth slow motion) and push on eggs at the physics rate.
  beginStep(action) {
    const d = this.d;
    const a = action.map((x) => clip(x, -1, 1));
    const ee0 = this.ee();
    for (let i = 0; i < 3; i++) this.eeTarget[i] = clip(this.eeTarget[i] + a[i] * V_MAX / CTRL_HZ, WS_LO[i], WS_HI[i]);
    // keep the target close to the real pinch point so the arm can't wind up
    for (let i = 0; i < 3; i++) this.eeTarget[i] = ee0[i] + clip(this.eeTarget[i] - ee0[i], -0.03, 0.03);
    this._ik(this.eeTarget).forEach((v, i) => { d.ctrl[i] = v; });
    d.ctrl[4] = ROLL;
    this._a = a; this._qCmd = gripToQ(a[3]);
    this._info = { grip: 0, shear: 0, slip: 0, event: null };
    this.sub = 0;
  }

  substep() {
    const d = this.d, qg = this.armQpos[5];
    // jaw servo: P loop with a softened gain and a torque limit, closed at the physics rate
    d.ctrl[this.gripAct] = clip(KP_G * (this._qCmd - d.qpos[qg]) - KD_G * d.qvel[this.gripDof], -TAU_LIM, TAU_LIM);
    this.mj.mj_step(this.m, d);
    const ti = this.touch(), info = this._info;
    info.grip += ti.grip / SUBSTEPS; info.shear += ti.shear / SUBSTEPS; info.slip += ti.slip / SUBSTEPS;
    this.sub++;
    this.live = ti;
    return ti;
  }

  endStep() {
    const info = this._info;
    info.event = this._checkEgg(info);   // shells crack under sustained load: judged on the 33 ms average
    this.prevAction = this._a; this.t++; this.last = info;
    return info;
  }

  step(action) {
    this.beginStep(action);
    for (let s = 0; s < SUBSTEPS; s++) this.substep();
    return this.endStep();
  }

  // put the arm back in the start pose without touching the eggs (between eggs in a full carton)
  resetArm() {
    const mj = this.mj, m = this.m, d = this.d;
    const q = this._ik(HOME, 200);
    q.forEach((v, i) => { d.qpos[this.armQpos[i]] = v; d.ctrl[i] = v; });
    d.qpos[this.armQpos[4]] = ROLL;
    d.qpos[this.armQpos[5]] = Q_OPEN;
    for (const a of this.allArmDofs) d.qvel[a] = 0;
    d.ctrl[4] = ROLL;
    mj.mj_forward(m, d);
    for (let i = 0; i < 60; i++) {
      d.ctrl[this.gripAct] = clip(KP_G * (Q_OPEN - d.qpos[this.armQpos[5]]), -TAU_LIM, TAU_LIM);
      mj.mj_step(m, d);
    }
    this.eeTarget = [...HOME]; this.prevAction = [0, 0, 0, 0];
    this.last = { grip: 0, shear: 0, slip: 0 };
  }

  _checkEgg(info) {
    const e = this.eggs[this.target];
    if (e.cracked) return null;
    e.peak = Math.max(e.peak, info.grip);
    if (info.grip > this.props(e.kind).crack) { this._crack(this.target); return 'crack'; }
    return null;
  }

  // swap the egg for yolk, white and shell pieces at the same place and velocity
  _crack(k) {
    const mj = this.mj, m = this.m, d = this.d, rng = this.rng;
    this.eggs[k].cracked = true;
    const p = this.eggPos(k);
    mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY.value, this.eggBody[k], this.v6a, 0);
    const v = Array.from(this.v6a.GetView());
    const offs = [[0, 0, -0.012], [0.004, -0.003, -0.02], [0.016, 0.01, -0.005], [-0.015, -0.008, -0.004]];
    this.debris[k].forEach((b, i) => {
      const j = m.body_jntadr[b], qa = m.jnt_qposadr[j], da = m.jnt_dofadr[j];
      d.qpos.set([p[0] + offs[i][0], p[1] + offs[i][1], p[2] + offs[i][2], 1, 0, 0, 0], qa);
      for (let c = 0; c < 3; c++) d.qvel[da + c] = v[3 + c] + (i ? rng.normal(0, 0.15) : 0);
      m.body_gravcomp[b] = 0;
      m.body_contype[b] = 1; m.body_conaffinity[b] = 1;      // broad-phase flags are cached per body
      for (const g of this._geomsOf(b)) { m.geom_contype[g] = 1; m.geom_conaffinity[g] = 1; }
    });
    m.body_gravcomp[this.eggBody[k]] = 1;
    m.geom_contype[this.eggGeom[k]] = 0; m.geom_conaffinity[this.eggGeom[k]] = 0;
    d.qpos.set([PARK[0] + k * 0.2, PARK[1] - 1, PARK[2]], this.eggQadr[k]);
    for (let c = 0; c < 6; c++) d.qvel[this.eggDadr[k] + c] = 0;
    mj.mj_forward(m, d);
  }

  // 'cracked' | 'basket' | 'dropped' | 'carton' | 'held'
  eggStatus(k = this.target) {
    const e = this.eggs[k];
    if (e.cracked) return 'cracked';
    const p = this.eggPos(k);
    if (Math.abs(p[0] - BASKET_C[0]) < BASKET_HALF && Math.abs(p[1] - BASKET_C[1]) < BASKET_HALF && p[2] < 0.07)
      return 'basket';
    const s = SLOTS[e.slot];
    if (Math.hypot(p[0] - s[0], p[1] - s[1]) < 0.02 && p[2] < 0.05) return 'carton';
    if (p[2] < EGG_H + 0.012) return 'dropped';
    return 'held';
  }
}

// Gaussian elimination with partial pivoting on a small dense system
function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}
