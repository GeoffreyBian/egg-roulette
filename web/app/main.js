import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import loadMujoco from '@mujoco/mujoco';
import { DROP_SPOTS, EggEnv, HOME, KINDS, PICK_ORDER, Rng, SUBSTEPS, CTRL_HZ } from '../sim/env.js';
import { Policy, Runner } from '../sim/policy.js';
import { ForceChart } from './chart.js';
import { SceneView, eggContacts, makeTextures } from './view.js';

const $ = (id) => document.getElementById(id);
const MAX_STEPS = 330;               // policy.MAX_STEPS: 11 s per egg
const COND = {
  A: { name: 'Robot demos only', blurb: 'Teleoperated robot demos. Operators squeeze by eye.' },
  B: { name: '+ human video', blurb: 'Plus 150 human demos seen on video: motion, no force.' },
  C: { name: '+ human touch demos', blurb: 'Plus 150 demos from a person in a touch glove. Robot has pad sensors.' },
  D: { name: 'Touch sensor, robot demos', blurb: 'Pad sensors on the robot, no human data.' },
};
const KIND_LABEL = { raw: 'raw', stone: 'stone', raw_fragile: 'fragile', medium: 'boiled', stone_light: 'light stone', stone_heavy: 'heavy stone', custom: 'custom' };
const OUT_LABEL = { success: 'intact', cracked: 'cracked', dropped: 'dropped', timeout: 'stalled' };
const CARTON_NOTE = {
  mystery: 'Each egg is raw (60 g, cracks above 2.2 N) or painted stone (300 g, slips out below about 3.5 N). They look identical, so the robot has to feel the difference.',
  raw: 'Six raw eggs: 60 g, cracking above 2.2 N of squeeze.',
  stone: 'Six painted stones: 300 g. The SO-101 needs about 3.5 N of squeeze to carry one.',
  unseen: 'Eggs no policy trained on: fragile (60 g, cracks at 1.6 N), hard-boiled (150 g, 3.5 N), light stone (180 g), heavy stone (400 g). Every policy here only ever saw raw and stone.',
  custom: 'Your own egg, never seen in training. Above about 350 g the jaw\'s torque limit can\'t hold it whatever the policy does; a crack threshold under the ~1 N needed to lift a raw egg is impossible too.',
};

// ---------------------------------------------------------------- settings (mirrored in the URL hash)
const S = {
  cond: 'C', n: 25, variant: '', compare: true, cmp: 'A', carton: 'mystery', mass: 120, crack: 3.0, hard: false,
  reveal: false, mu: 0.8, noise: 3, src: 'pads', seed: 1, speed: 1,
  contacts: true, sensors: false, collision: false,
};
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of h) {
    if (!(k in S)) continue;
    S[k] = typeof S[k] === 'number' ? Number(v) : typeof S[k] === 'boolean' ? v === '1' : v;
  }
}
function writeHash() {
  const keys = ['cond', 'n', 'variant', 'compare', 'cmp', 'carton', 'mass', 'crack', 'hard', 'mu', 'noise', 'src', 'seed'];
  const h = new URLSearchParams();
  for (const k of keys) h.set(k, typeof S[k] === 'boolean' ? (S[k] ? '1' : '0') : S[k]);
  history.replaceState(null, '', `#${h}`);
}

// ---------------------------------------------------------------- loading
let mj, index, tex;
const policies = new Map();
const models = [];

function progress(msg, frac) { $('ld-msg').textContent = msg; $('ld-bar').style.width = `${Math.round(100 * frac)}%`; }

async function boot() {
  readHash();
  progress('Loading MuJoCo 3.17 (WebAssembly)…', 0.05);
  mj = await loadMujoco();
  progress('Loading the SO-101 and the scene…', 0.45);
  const xml = await (await fetch('assets/scene.xml')).text();
  const files = [...new Set([...xml.matchAll(/<mesh [^>]*file="([^"]+)"/g)].map((m) => m[1]))];
  mj.FS.mkdir('/w'); mj.FS.mkdir('/w/meshes');
  let done = 0;
  await Promise.all(files.map(async (f) => {
    const buf = await (await fetch(`assets/meshes/${f}`)).arrayBuffer();
    mj.FS.writeFile(`/w/meshes/${f}`, new Uint8Array(buf));
    progress('Loading the SO-101 and the scene…', 0.45 + 0.35 * (++done / files.length));
  }));
  mj.FS.writeFile('/w/scene.xml', xml);
  progress('Compiling the model…', 0.82);
  await new Promise((r) => setTimeout(r, 0));
  // one model per side: each run edits model fields (egg masses and sizes, crack debris contacts)
  models.push(mj.MjModel.from_xml_path('/w/scene.xml'), mj.MjModel.from_xml_path('/w/scene.xml'));
  index = await (await fetch('assets/policies/index.json')).json();
  tex = makeTextures({ wood: 'assets/wood.jpg', pulp: 'assets/pulp.jpg' });
  progress('Loading the policies…', 0.9);
  initThree();
  buildPanel();
  await restart();
  $('loading').classList.add('done');
  requestAnimationFrame(frame);
}

async function getPolicy(name) {
  if (!policies.has(name)) {
    const meta = index.find((p) => p.name === name);
    policies.set(name, await Policy.load(meta, `assets/policies/${name}.bin`));
  }
  return policies.get(name);
}

const policyName = (cond, n, variant) => (variant && cond === 'C' && n === 25 ? `var_${variant}` : `${cond}_n${n}`);

// ---------------------------------------------------------------- a carton run for one policy
class Session {
  constructor(side, model) {
    this.side = side;
    this.env = new EggEnv(mj, model, {});
    this.view = new SceneView(model, tex);
    this.view.scene.environment = envMap;
    this.replay = new mj.MjData(model);
    this.vel = new mj.DoubleBuffer(6);
    this.eggBodies = new Set(this.env.eggBody.concat(this.env.debris.flat()));
  }

  start(policy, label, color, kinds, seed) {
    const env = this.env;
    this.policy = policy; this.label = label; this.color = color;
    env.rng = new Rng(seed);
    env.visionNoise = S.noise / 1000;
    env.setFriction(S.mu);
    env.reset(kinds);
    env.sub = undefined;                          // no control step in flight
    this.runner = new Runner(policy, { touchSource: policy.meta.touch ? S.src : 'pads' });
    this.kinds = kinds;
    this.frames = [];
    this.outcomes = [];
    this.i = 0;
    this.phase = 'policy';
    this.startEgg();
    this.record(null);
  }

  startEgg() {
    const k = PICK_ORDER[this.i];
    this.env.setTarget(k, DROP_SPOTS[this.i]);
    this.runner.reset();
    this.obs = this.env.obs();
    this.steps = 0; this.basket = 0; this.scripted = 0;
  }

  get done() { return this.phase === 'done'; }

  // one physics step (2 ms); control decisions happen every SUBSTEPS physics steps (30 Hz)
  tick() {
    if (this.done) return null;
    const env = this.env;
    if (env.sub === undefined || env.sub >= SUBSTEPS) {
      let a;
      if (this.phase === 'policy') a = this.action = this.runner.act(this.obs);
      else if (this.phase === 'retreat') a = [0, 0, 0.6, -1];
      else {
        const ee = env.ee();
        a = [...HOME.map((h, j) => Math.max(-1, Math.min(1, (h - ee[j]) / 0.0073))), -1];
      }
      env.beginStep(a);
    }
    this.perturb();
    env.substep();
    if (env.sub < SUBSTEPS) return null;
    const info = env.endStep();
    return this.post(info);
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
        this.outcomes.push({ kind: e.kind, out, peak: e.peak, t: this.steps / CTRL_HZ });
        this.phase = 'retreat';
        event = out;
      }
    } else if (this.phase === 'retreat' && ++this.scripted >= 22) {
      this.phase = 'home'; this.scripted = 0;
    } else if (this.phase === 'home' && ++this.scripted >= 36) {
      env.resetArm();
      if (++this.i < 6) { this.phase = 'policy'; this.startEgg(); } else this.phase = 'done';
    }
    this.record(info);
    return event;
  }

  record(info) {
    const env = this.env;
    this.frames.push({
      qpos: Float64Array.from(env.d.qpos),
      egg: Math.min(this.i, 5), phase: this.phase, steps: this.steps,
      grip: info ? info.grip : 0,
      pads: env.nativeTouch(),
      contacts: S.contacts ? eggContacts(env, this.eggBodies) : [],
      outcomes: this.outcomes.length,
    });
  }

  // the mouse spring: pulls the grabbed body toward the cursor at the physics rate
  perturb() {
    const d = this.env.d;
    if (!grab || grab.session !== this) return;
    const b = grab.body, m = this.env.m, mass = m.body_mass[b];
    const p = d.xpos.subarray(3 * b, 3 * b + 3);
    mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY.value, b, this.vel, 0);
    const v = this.vel.GetView();
    const k = mass * 900, c = 2 * mass * 30;       // ~30 rad/s spring, critically damped
    for (let j = 0; j < 3; j++) d.xfrc_applied[6 * b + j] = k * (grab.target[j] - p[j]) - c * v[3 + j] + (j === 2 ? mass * 9.81 : 0);
  }

  release(b) { for (let j = 0; j < 6; j++) this.env.d.xfrc_applied[6 * b + j] = 0; }

  // pose the view from the live state, or from a recorded frame
  show(f = null) {
    const m = this.env.m;
    let d = this.env.d, pads = this.env.nativeTouch(), contacts = null;
    if (f !== null) {
      const fr = this.frames[Math.min(f, this.frames.length - 1)];
      this.replay.qpos.set(fr.qpos);
      mj.mj_kinematics(m, this.replay);
      d = this.replay; pads = fr.pads; contacts = fr.contacts;
    } else if (S.contacts) contacts = eggContacts(this.env, this.eggBodies);
    this.view.update(d, { sensors: S.sensors, padForces: pads });
    this.view.setContacts(S.contacts ? contacts : []);
    const gb = grab && grab.session === this && f === null ? grab : null;
    this.view.setSpring(gb ? Array.from(d.xpos.subarray(3 * gb.body, 3 * gb.body + 3)) : null, gb ? gb.target : null);
  }

  // squeeze trace of the egg showing at frame f
  trace(f) {
    const F = this.frames, end = Math.min(f, F.length - 1), egg = F[end].egg, t = [], g = [];
    let s = end;
    while (s > 0 && F[s - 1].egg === egg && F[s - 1].phase === 'policy') s--;
    for (let j = s; j <= end; j++) if (F[j].phase === 'policy' || (j > s && F[j - 1].phase === 'policy')) { t.push(F[j].steps / CTRL_HZ); g.push(F[j].grip); }
    return { t, f: g };
  }
}

// ---------------------------------------------------------------- three.js
let renderer, camera, controls, envMap, chart;
const sessions = [];
let grab = null, poke = false;
const CAMS = {
  hero: { pos: [0.52, 0.30, 0.34], target: [0.17, 0.02, 0.03] },
  front: { pos: [0.64, 0.33, 0.02], target: [0.16, 0.0, 0.04] },
  top: { pos: [0.19, 0.6, 0.06], target: [0.19, 0.0, 0.0] },
  follow: null,
};
let camMode = 'hero', lastFit = 1;

function initThree() {
  const canvas = $('gl');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setScissorTest(true);
  const pm = new THREE.PMREMGenerator(renderer);
  envMap = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.minDistance = 0.12; controls.maxDistance = 1.6;
  controls.maxPolarAngle = Math.PI * 0.495;
  setCam('hero', true);
  for (const model of models) sessions.push(new Session(sessions.length, model));
  for (const s of sessions) s.view.scene.environmentIntensity = 0.35;
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  chart = new ForceChart($('chart'), $('tt'));
}

function setCam(name, instant = false) {
  camMode = name;
  document.querySelectorAll('#cams button').forEach((b) => b.classList.toggle('on', b.dataset.cam === name));
  const c = CAMS[name];
  if (!c) return;
  const target = new THREE.Vector3(...c.target);
  const pos = new THREE.Vector3(...c.pos).sub(target).multiplyScalar(lastFit).add(target);
  camTween = { pos, target, k: instant ? 1 : 0.12 };
}
let camTween = null;

// viewport rectangles: one view, or two side by side (stacked when the screen is tall)
function layout() {
  const el = $('viewport'), W = el.clientWidth, H = el.clientHeight;
  const n = S.compare ? 2 : 1, vertical = n === 2 && H > W * 0.9;
  const rects = [];
  for (let i = 0; i < n; i++) {
    rects.push(vertical ? { x: 0, y: i * H / 2, w: W, h: H / 2 } : { x: i * W / n, y: 0, w: W / n, h: H });
  }
  return { W, H, rects, vertical };
}

function rectAt(e) {
  const { rects } = layout();
  return rects.findIndex((r) => e.offsetX >= r.x && e.offsetX < r.x + r.w && e.offsetY >= r.y && e.offsetY < r.y + r.h);
}

const ray = new THREE.Raycaster();
function pointerRay(e, i) {
  const r = layout().rects[i];
  const ndc = new THREE.Vector2(((e.offsetX - r.x) / r.w) * 2 - 1, -((e.offsetY - r.y) / r.h) * 2 + 1);
  camera.aspect = r.w / r.h; camera.updateProjectionMatrix();
  ray.setFromCamera(ndc, camera);
  return ray;
}
const toMj = (v) => [v.x, -v.z, v.y];     // three.js y-up -> MuJoCo z-up

function onDown(e) {
  if (!(e.shiftKey || poke) || scrubbing()) return;
  const i = rectAt(e);
  if (i < 0) return;
  const s = sessions[i], r = pointerRay(e, i);
  const hit = r.intersectObjects(s.view.pickable.filter((o) => o.visible), false)[0];
  if (!hit) return;
  const normal = camera.getWorldDirection(new THREE.Vector3()).negate();
  grab = { session: s, body: hit.object.userData.body, plane: new THREE.Plane().setFromNormalAndCoplanarPoint(normal, hit.point), target: toMj(hit.point), side: i };
  controls.enabled = false;
  $('gl').setPointerCapture(e.pointerId);
  if (!running) setRunning(true);
}
function onMove(e) {
  if (!grab) return;
  const p = new THREE.Vector3();
  if (pointerRay(e, grab.side).ray.intersectPlane(grab.plane, p)) {
    const t = toMj(p);
    t[2] = Math.max(t[2], 0.0);
    grab.target = t;
  }
}
function onUp() {
  if (!grab) return;
  grab.session.release(grab.body);
  grab = null;
  controls.enabled = true;
}

// ---------------------------------------------------------------- run control
let running = true, budget = 0, last = performance.now(), viewFrame = null;
const scrubbing = () => viewFrame !== null;

function setRunning(on) {
  running = on;
  $('play').classList.toggle('paused', !on);
  $('play').setAttribute('aria-label', on ? 'Pause' : 'Play');
  if (on) { viewFrame = null; budget = 0; last = performance.now(); }
}

function cartonKinds(rng) {
  if (S.carton === 'raw') return Array(6).fill('raw');
  if (S.carton === 'stone') return Array(6).fill('stone');
  if (S.carton === 'unseen') {
    const k = ['raw_fragile', 'medium', 'stone_light', 'stone_heavy', 'raw_fragile', 'medium'];
    for (let i = k.length - 1; i > 0; i--) { const j = rng.int(i + 1); [k[i], k[j]] = [k[j], k[i]]; }
    return k;
  }
  if (S.carton === 'custom') return Array(6).fill({ kind: 'custom', mass: S.mass / 1000, crack: S.hard ? Infinity : S.crack });
  return Array.from({ length: 6 }, () => (rng.uniform() < 0.5 ? 'raw' : 'stone'));
}

let runId = 0;
async function restart() {
  const id = ++runId;
  writeHash();
  const main = policyName(S.cond, S.n, S.variant);
  const pols = [await getPolicy(main)];
  if (S.compare) pols.push(await getPolicy(policyName(S.cmp, S.n, '')));
  if (id !== runId) return;
  const kinds = cartonKinds(new Rng(S.seed * 7919 + 13));
  const labels = [policyLabel(S.cond, S.variant), policyLabel(S.cmp, '')];
  const colors = [css(`--c-${S.cond}`), css(`--c-${S.cmp}`)];
  if (grab) onUp();
  sessions.forEach((s, i) => {
    if (i < pols.length) s.start(pols[i], `${labels[i]} · ${S.n} robot demos`, colors[i], kinds, S.seed);
  });
  for (const s of sessions) s.view.showCollision(S.collision);
  $('scrub').value = 0;
  setRunning(true);
  updateLabels(true);
  renderScores();
}

function policyLabel(cond, variant) {
  if (variant && cond === 'C' && S.n === 25) return index.find((p) => p.variant === variant).label.replace(/^./, (c) => c.toUpperCase());
  return COND[cond].name.replace(/^\+ /, '+ ');
}
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const active = () => sessions.slice(0, S.compare ? 2 : 1);

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (running && !scrubbing()) {
    budget += dt * 500 * S.speed;
    const n = Math.min(Math.floor(budget), 200);
    budget -= n;
    for (let k = 0; k < n; k++) {
      for (const s of active()) {
        const ev = s.tick();
        if (ev) { toast(s, ev); renderScores(); }
      }
    }
    if (active().every((s) => s.done)) { setRunning(false); renderScores(); }
    const len = Math.max(...active().map((s) => s.frames.length)) - 1;
    $('scrub').max = len; $('scrub').value = len;
  }
  draw();
  requestAnimationFrame(frame);
}

function draw() {
  const { W, H, rects, vertical } = layout();
  const dpr = renderer.getPixelRatio();
  if (renderer.domElement.width !== Math.round(W * dpr) || renderer.domElement.height !== Math.round(H * dpr)) renderer.setSize(W, H, false);
  const act = active();
  // camera: tween to presets, or follow the pinch point
  if (camMode === 'follow') {
    const ee = act[0].env.ee();
    const tgt = new THREE.Vector3(ee[0], ee[2] - 0.01, -ee[1]);
    const off = camera.position.clone().sub(controls.target);
    if (off.length() > 0.3) off.setLength(0.24);
    controls.target.lerp(tgt, 0.15);
    camera.position.copy(controls.target).add(off);
  } else if (camTween) {
    controls.target.lerp(camTween.target, camTween.k);
    camera.position.lerp(camTween.pos, camTween.k);
    if (camera.position.distanceTo(camTween.pos) < 0.002) camTween = null;
  }
  controls.update();
  // narrow views (split screen, phones) need the camera further back to keep the carton and basket in frame
  const aspect = rects[0].w / rects[0].h, fit = Math.min(1.75, Math.max(1, 1.45 / aspect));
  if (Math.abs(fit - lastFit) > 0.01) {
    const off = camera.position.clone().sub(controls.target).multiplyScalar(fit / lastFit);
    camera.position.copy(controls.target).add(off);
    if (camTween) camTween.pos = camTween.target.clone().add(camTween.pos.clone().sub(camTween.target).multiplyScalar(fit / lastFit));
    lastFit = fit;
  }
  const f = scrubbing() ? viewFrame : null;
  act.forEach((s, i) => {
    s.show(f);
    const r = rects[i];
    camera.aspect = r.w / r.h; camera.updateProjectionMatrix();
    renderer.setViewport(r.x, H - r.y - r.h, r.w, r.h);
    renderer.setScissor(r.x, H - r.y - r.h, r.w, r.h);
    renderer.render(s.view.scene, camera);
  });
  // labels and the chart
  updateLabels(false, rects, vertical);
  const fr = f ?? Math.max(...act.map((s) => s.frames.length - 1));
  chart.series = act.map((s) => ({ label: s.label.split(' · ')[0], color: s.color, ...s.trace(Math.min(fr, s.frames.length - 1)) }));
  chart.cursor = null;
  chart.draw();
  const t = fr / CTRL_HZ;
  $('time').textContent = `${t.toFixed(1)} s`;
  if (scrubbing()) renderScores(f);
}

// ---------------------------------------------------------------- overlays
let labelEls = [];
function updateLabels(rebuild, rects = layout().rects) {
  const host = $('labels'), act = active();
  if (rebuild || labelEls.length !== act.length) {
    host.innerHTML = '';
    labelEls = act.map((s) => {
      const el = document.createElement('div');
      el.className = 'label';
      el.innerHTML = `<span class="sw" style="background:${s.color}"></span><b></b><span class="f"></span>`;
      el.querySelector('b').textContent = s.label.split(' · ')[0];
      host.appendChild(el);
      return el;
    });
    if (act.length === 2) { const dv = document.createElement('div'); dv.className = 'divider'; host.appendChild(dv); }
  }
  act.forEach((s, i) => {
    const r = rects[i], el = labelEls[i];
    el.style.left = `${r.x + 12}px`; el.style.top = `${r.y + 12}px`;
    const fr = scrubbing() ? s.frames[Math.min(viewFrame, s.frames.length - 1)] : s.frames[s.frames.length - 1];
    const live = scrubbing() ? fr.grip : (s.env.live ? s.env.live.grip : 0);
    el.querySelector('.f').textContent = fr.phase === 'policy' ? `squeeze ${live.toFixed(2)} N` : fr.phase === 'done' ? 'carton done' : '';
  });
  const dv = host.querySelector('.divider');
  if (dv && rects.length === 2) {
    const v = rects[1].y > 0;
    Object.assign(dv.style, v ? { left: '0', right: '0', top: `${rects[1].y}px`, height: '1px', width: '' } : { top: '0', bottom: '0', left: `${rects[1].x}px`, width: '1px', height: '' });
  }
}

function toast(s, out) {
  const r = layout().rects[s.side];
  const o = s.outcomes[s.outcomes.length - 1];
  const color = { success: 'var(--good)', cracked: 'var(--bad)', dropped: 'var(--drop)', timeout: 'var(--warn)' }[out];
  const text = { success: 'Moved intact', cracked: 'Cracked', dropped: 'Dropped', timeout: 'Stalled' }[out];
  const why = out === 'cracked' ? `peak ${o.peak.toFixed(1)} N` : out === 'dropped' ? `it was ${kindName(o.kind)}` : out === 'success' ? `${o.t.toFixed(1)} s` : '11 s, gave up';
  const el = document.createElement('div');
  el.className = 'toast';
  el.style.background = color;
  el.style.left = `${r.x + r.w / 2}px`; el.style.top = `${r.y + r.h - 64}px`;
  el.innerHTML = `${text}<small>${why}</small>`;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), 2500);
}

const kindName = (k) => (typeof k === 'string' ? `a ${KIND_LABEL[k]} egg` : 'your egg');

function renderScores(f = null) {
  const host = $('scores');
  host.innerHTML = '';
  for (const s of active()) {
    const fr = f === null ? s.frames.length - 1 : Math.min(f, s.frames.length - 1);
    const nOut = s.frames[fr] ? s.frames[fr].outcomes : 0;
    const outs = s.outcomes.slice(0, nOut);
    const cur = s.frames[fr] ? s.frames[fr].egg : 0;
    const ok = outs.filter((o) => o.out === 'success').length;
    const row = document.createElement('div');
    row.className = 'srow';
    row.innerHTML = `<div class="sname"><span class="sw" style="background:${s.color}"></span>${s.label.split(' · ')[0]}<span class="tot">${ok}/${outs.length} intact</span></div><div class="cells"></div>`;
    const cells = row.querySelector('.cells');
    for (let i = 0; i < 6; i++) {
      const o = outs[i], c = document.createElement('div');
      const kind = s.kinds[PICK_ORDER[i]], kl = typeof kind === 'string' ? KIND_LABEL[kind] : 'custom';
      const showKind = o || S.reveal || S.carton !== 'mystery';
      c.className = `cell${o ? ` ${o.out}` : ''}${!o && i === cur && !s.done ? ' now' : ''}`;
      c.innerHTML = `<span class="k">${showKind ? kl : '?'}</span>${o ? `<span class="o">${OUT_LABEL[o.out]}</span>` : ''}`;
      c.title = o ? `${kl}: ${OUT_LABEL[o.out]}, peak squeeze ${o.peak.toFixed(2)} N` : showKind ? kl : 'unknown until picked';
      cells.appendChild(c);
    }
    host.appendChild(row);
  }
}

// ---------------------------------------------------------------- panel
function successOf(cond, n) {
  const p = index.find((q) => q.name === `${cond}_n${n}`);
  return p.success_seeds.reduce((a, b) => a + b, 0) / p.success_seeds.length;
}

function buildPanel() {
  // conditions
  const conds = $('conds');
  for (const c of 'ABCD') {
    const b = document.createElement('button');
    b.className = 'cond'; b.dataset.cond = c;
    b.innerHTML = `<span class="sw" style="background:var(--c-${c})"></span><b>${COND[c].name}</b><span class="pct"></span><small>${COND[c].blurb}</small>`;
    b.onclick = () => { S.cond = c; if (S.cmp === c) S.cmp = c === 'A' ? 'C' : 'A'; syncPanel(); restart(); };
    conds.appendChild(b);
  }
  document.querySelectorAll('#nrobot button').forEach((b) => { b.onclick = () => { S.n = +b.dataset.n; syncPanel(); restart(); }; });
  const vs = $('variant');
  vs.innerHTML = '<option value="">Standard (the headline policy)</option>' +
    index.filter((p) => p.variant).map((p) => `<option value="${p.variant}">${p.label}</option>`).join('');
  vs.onchange = () => { S.variant = vs.value; syncPanel(); restart(); };
  const cc = $('compare-cond');
  cc.innerHTML = 'ABCD'.split('').map((c) => `<option value="${c}">${COND[c].name}</option>`).join('');
  cc.onchange = () => { S.cmp = cc.value; restart(); };
  $('compare').onchange = (e) => { S.compare = e.target.checked; syncPanel(); restart(); };
  // carton
  document.querySelectorAll('#carton button').forEach((b) => { b.onclick = () => { S.carton = b.dataset.carton; syncPanel(); restart(); }; });
  const live = (id, key, fmt, scale = 1) => {
    const el = $(id);
    el.oninput = () => { S[key] = +el.value * scale; $(`${id}-o`).textContent = fmt(S[key]); };
    el.onchange = () => restart();
  };
  live('c-mass', 'mass', (v) => `${v} g`);
  live('c-crack', 'crack', (v) => `${v.toFixed(1)} N`);
  $('c-hard').onchange = (e) => { S.hard = e.target.checked; syncPanel(); restart(); };
  $('reveal').onchange = (e) => { S.reveal = e.target.checked; renderScores(); };
  live('mu', 'mu', (v) => v.toFixed(2));
  live('noise', 'noise', (v) => `${v.toFixed(1)} mm`);
  document.querySelectorAll('#touchsrc button').forEach((b) => { b.onclick = () => { S.src = b.dataset.src; syncPanel(); restart(); }; });
  // view toggles
  $('v-contacts').onchange = (e) => { S.contacts = e.target.checked; };
  $('v-sensors').onchange = (e) => { S.sensors = e.target.checked; };
  $('v-collision').onchange = (e) => { S.collision = e.target.checked; for (const s of sessions) s.view.showCollision(S.collision); };
  // transport
  $('play').onclick = () => {
    if (active().every((s) => s.done) && !scrubbing()) { restart(); return; }
    setRunning(!running);
  };
  $('again').onclick = () => restart();
  $('fresh').onclick = () => { S.seed = 1 + Math.floor(Math.random() * 99999); restart(); };
  document.querySelectorAll('#speed button').forEach((b) => {
    b.onclick = () => { S.speed = +b.dataset.speed; document.querySelectorAll('#speed button').forEach((x) => x.classList.toggle('on', x === b)); };
  });
  $('scrub').oninput = (e) => {
    const v = +e.target.value, max = +e.target.max;
    if (running) setRunning(false);
    viewFrame = v >= max && active().some((s) => !s.done) ? null : v;
    if (v >= max) viewFrame = active().every((s) => s.done) ? v : null;
  };
  document.querySelectorAll('#cams button').forEach((b) => { b.onclick = () => setCam(b.dataset.cam); });
  $('poke').onclick = () => {
    poke = !poke;
    $('poke').setAttribute('aria-pressed', poke);
    controls.enableRotate = !poke;
  };
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  });
  drawGain();
  syncPanel();
  const parity = 'A parity test (web/test/parity.mjs) checks this code against Python: fed the same inputs, the ' +
    'policy actions match to 2e-7, replayed Python episodes all end the same way, and on 60 fresh eggs the browser ' +
    'scores 85% for the touch policy and 50% for robot demos only (Python: 92% and 45%).';
  $('parity').textContent = parity;
}

function syncPanel() {
  document.querySelectorAll('.cond').forEach((b) => {
    b.classList.toggle('on', b.dataset.cond === S.cond);
    b.querySelector('.pct').textContent = `${Math.round(100 * successOf(b.dataset.cond, S.n))}%`;
  });
  document.querySelectorAll('#nrobot button').forEach((b) => b.classList.toggle('on', +b.dataset.n === S.n));
  const vOk = S.cond === 'C' && S.n === 25;
  $('variant').disabled = !vOk;
  if (!vOk) S.variant = '';
  $('variant').value = S.variant;
  $('variant').title = vOk ? '' : 'Recipe variants exist for the touch policy with 25 robot demos';
  $('compare').checked = S.compare;
  $('compare-cond').value = S.cmp;
  $('compare-cond').disabled = !S.compare;
  [...$('compare-cond').options].forEach((o) => { o.disabled = o.value === S.cond; });
  document.querySelectorAll('#carton button').forEach((b) => b.classList.toggle('on', b.dataset.carton === S.carton));
  $('carton-note').textContent = CARTON_NOTE[S.carton];
  $('custom').hidden = S.carton !== 'custom';
  for (const [id, key, fmt] of [['c-mass', 'mass', (v) => `${v} g`], ['c-crack', 'crack', (v) => `${v.toFixed(1)} N`],
    ['mu', 'mu', (v) => v.toFixed(2)], ['noise', 'noise', (v) => `${v.toFixed(1)} mm`]]) {
    $(id).value = S[key]; $(`${id}-o`).textContent = fmt(S[key]);
  }
  $('c-hard').checked = S.hard; $('c-crack').disabled = S.hard;
  const touchPolicy = S.cond === 'C' || S.cond === 'D';
  document.querySelectorAll('#touchsrc button').forEach((b) => { b.classList.toggle('on', b.dataset.src === S.src); b.disabled = !touchPolicy; });
  const meta = index.find((p) => p.name === policyName(S.cond, S.n, S.variant));
  const notes = [];
  if (S.mu !== 0.8) notes.push(`Trained at μ = 0.80; the robustness grid in the README shows how far each policy stretches.`);
  if (!touchPolicy) notes.push('The policy on the left has no touch input, so the touch setting does nothing for it.');
  else if (S.src === 'servo') notes.push('Servo current gives a rough squeeze estimate and no load or slip. The glove-trained policy was never shown it: 5% in the ablations.');
  else if (S.src === 'off') notes.push('Touch channels read zero, as if the pad sensors were unplugged.');
  if (meta && meta.proxy) notes.push('This recipe was trained on the servo-current estimate, so it always reads that.');
  $('test-note').innerHTML = notes.join(' ');
  $('test-note').hidden = !notes.length;
  const lo = Math.min(...meta.success_seeds), hi = Math.max(...meta.success_seeds);
  $('gain-cap').innerHTML = `Python evaluation, 5 seeds × 60 eggs each. Loaded: seed ${meta.seed}, <b>${Math.round(100 * meta.success)}%</b>` +
    ` (seeds span ${Math.round(100 * lo)}–${Math.round(100 * hi)}%).`;
  drawGain();
}

function drawGain() {
  const svg = $('gain'), W = 320, H = 150, L = 30, R = 8, T = 8, B = 22;
  const ns = [10, 25, 50, 100], x = (n) => L + (Math.log(n / 10) / Math.log(10)) * (W - L - R), y = (p) => T + (1 - p) * (H - T - B);
  let s = '';
  for (const p of [0, 0.5, 1]) s += `<line class="gl" x1="${L}" x2="${W - R}" y1="${y(p)}" y2="${y(p)}"/><text x="${L - 6}" y="${y(p) + 3}" text-anchor="end">${p * 100}%</text>`;
  for (const n of ns) s += `<text x="${x(n)}" y="${H - 6}" text-anchor="middle">${n}</text>`;
  s += `<text x="${W - R}" y="${H - 6}" text-anchor="end" dx="0" dy="-12" style="font-size:9px">robot demos</text>`;
  for (const c of 'ABCD') {
    const on = c === S.cond;
    const pts = ns.map((n) => [x(n), y(successOf(c, n))]);
    s += `<polyline points="${pts.map((p) => p.join(',')).join(' ')}" fill="none" stroke="var(--c-${c})" stroke-width="${on ? 2.5 : 1.5}" opacity="${on ? 1 : 0.55}"/>`;
    ns.forEach((n, i) => {
      const sel = on && n === S.n;
      s += `<circle class="pt" data-c="${c}" data-n="${n}" cx="${pts[i][0]}" cy="${pts[i][1]}" r="${sel ? 5.5 : 3.5}" fill="${sel ? 'var(--c-' + c + ')' : 'var(--panel)'}" stroke="var(--c-${c})" stroke-width="2"><title>${COND[c].name}, ${n} robot demos: ${Math.round(100 * successOf(c, n))}%</title></circle>`;
    });
  }
  svg.innerHTML = s;
  svg.querySelectorAll('.pt').forEach((p) => {
    p.onclick = () => { S.cond = p.dataset.c; S.n = +p.dataset.n; if (S.cmp === S.cond) S.cmp = S.cond === 'A' ? 'C' : 'A'; syncPanel(); restart(); };
  });
}

boot().catch((err) => {
  console.error(err);
  $('loading').classList.add('error');
  progress(`Couldn't start the simulator: ${err.message}. It needs a browser with WebAssembly and WebGL.`, 1);
});
