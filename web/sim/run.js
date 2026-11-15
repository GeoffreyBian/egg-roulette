// One carton run: a policy moves six eggs, one after another, the way eggsim/render.py's run_carton does
// (policy episode, rise, go home, reset the arm). Runs in a Web Worker in the browser (app/worker.js).
import { DROP_SPOTS, EggEnv, HOME, PICK_ORDER, Rng, SUBSTEPS, CTRL_HZ } from './env.js';
import { PixelPolicy, PixelRunner, Runner } from './policy.js';
import { Cameras } from './camera.js';

const MAX_STEPS = 330;               // policy.MAX_STEPS: 11 s per egg

export class CartonRun {
  constructor(mj, model, camSpec) {
    this.mj = mj;
    this.env = new EggEnv(mj, model, {});
    this.cams = new Cameras(mj, model, camSpec);
    this.vel = new mj.DoubleBuffer(6);
    this.eggBodies = new Set(this.env.eggBody.concat(this.env.debris.flat()));
    this.grab = null;
    this.phase = 'done';
  }

  // opts: noise (mm), mu, src (touch source), saliency, contacts
  start(policy, kinds, seed, opts) {
    const env = this.env;
    this.opts = opts;
    env.rng = new Rng(seed);
    env.visionNoise = opts.noise / 1000;
    env.setFriction(opts.mu);
    env.reset(kinds);
    env.sub = undefined;                          // no control step in flight
    const src = policy.meta.touch ? opts.src : 'pads';
    this.pixels = policy instanceof PixelPolicy;
    this.runner = this.pixels ? new PixelRunner(policy, this.cams, { touchSource: src }) : new Runner(policy, { touchSource: src });
    this.sal = null; this.view = null;
    this.outcomes = [];
    this.i = 0;
    this.phase = 'policy';
    this.grab = null;
    this.startEgg();
    return this.record(null);
  }

  startEgg() {
    this.env.setTarget(PICK_ORDER[this.i], DROP_SPOTS[this.i]);
    this.runner.reset();
    this.obs = this.env.obs();
    this.steps = 0; this.basket = 0; this.scripted = 0;
  }

  get done() { return this.phase === 'done'; }

  // one physics step (2 ms). Returns a record when a control step (30 Hz) completes, else null.
  tick() {
    if (this.done) return null;
    const env = this.env;
    if (env.sub === undefined || env.sub >= SUBSTEPS) {
      let a;
      this.view = null;
      if (this.phase === 'policy') {
        a = this.action = this.pixels ? this.runner.act(this.obs, env.d) : this.runner.act(this.obs);
        if (this.pixels) {
          // gradient saliency is one backward pass per action output, about 5 control steps of compute: refresh it twice a second
          if (this.opts.saliency && (this.steps % 15 === 0 || !this.sal)) this.sal = this.runner.pol.saliency();
          this.view = { frames: this.runner.frames, maps: this.opts.saliency ? this.sal : null };
        }
      } else if (this.phase === 'retreat') a = [0, 0, 0.6, -1];
      else {
        const ee = env.ee();
        a = [...HOME.map((h, j) => Math.max(-1, Math.min(1, (h - ee[j]) / 0.0073))), -1];
      }
      env.beginStep(a);
    }
    this.perturb();
    env.substep();
    if (env.sub < SUBSTEPS) return null;
    return this.post(env.endStep());
  }

  post(info) {
    const env = this.env;
    let event = null;
    if (this.phase === 'policy') {
      this.steps++;
      this.obs = env.obs();
      const st = env.eggStatus();
      let out = null;
      if (st === 'cracked' || st === 'dropped') out = st;
      else if (st === 'basket' && this.action[3] < 0 && ++this.basket > 8) out = 'success';
      else if (this.steps >= MAX_STEPS) out = 'timeout';
      const e = env.eggs[env.target];
      if (out) {
        event = { kind: e.kind, out, peak: e.peak, t: this.steps / CTRL_HZ };
        this.outcomes.push(event);
        this.phase = 'retreat';
      }
    } else if (this.phase === 'retreat' && ++this.scripted >= 22) {
      this.phase = 'home'; this.scripted = 0;
    } else if (this.phase === 'home' && ++this.scripted >= 36) {
      env.resetArm();
      if (++this.i < 6) { this.phase = 'policy'; this.startEgg(); } else this.phase = 'done';
    }
    return this.record(info, event);
  }

  record(info, event = null) {
    const env = this.env;
    return {
      qpos: Float64Array.from(env.d.qpos),
      egg: Math.min(this.i, 5), phase: this.phase, steps: this.steps,
      grip: info ? info.grip : 0,
      pads: env.nativeTouch(),
      contacts: this.opts.contacts ? env.contacts(this.eggBodies) : [],
      outcomes: this.outcomes.length, event,
      view: this.view,               // pixel policies: the two camera images and the saliency maps
    };
  }

  // the mouse spring: pulls the grabbed body toward the cursor at the physics rate
  perturb() {
    if (!this.grab) return;
    const mj = this.mj, d = this.env.d, m = this.env.m, b = this.grab.body, mass = m.body_mass[b];
    const p = d.xpos.subarray(3 * b, 3 * b + 3);
    mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY.value, b, this.vel, 0);
    const v = this.vel.GetView();
    const k = mass * 900, c = 2 * mass * 30;       // ~30 rad/s spring, critically damped
    for (let j = 0; j < 3; j++) d.xfrc_applied[6 * b + j] = k * (this.grab.target[j] - p[j]) - c * v[3 + j] + (j === 2 ? mass * 9.81 : 0);
  }

  release() {
    if (!this.grab) return;
    for (let j = 0; j < 6; j++) this.env.d.xfrc_applied[6 * this.grab.body + j] = 0;
    this.grab = null;
  }
}
