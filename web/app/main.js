import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import loadMujoco from '@mujoco/mujoco';
import { PICK_ORDER, Rng, CTRL_HZ } from '../sim/env.js';
import { Cameras } from '../sim/camera.js';
import { ForceChart } from './chart.js';
import { SceneView, makeTextures } from './view.js';

const $ = (id) => document.getElementById(id);
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
  input: 'pixels', cond: 'C', n: 25, variant: '', compare: true, cmp: 'A', carton: 'mystery', mass: 120, crack: 3.0, hard: false,
  reveal: false, mu: 0.8, noise: 3, src: 'pads', seed: 1, speed: 1,
  contacts: true, sensors: false, collision: false, saliency: true,
};
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of h) {
    if (!(k in S)) continue;
    S[k] = typeof S[k] === 'number' ? Number(v) : typeof S[k] === 'boolean' ? v === '1' : v;
  }
}
function writeHash() {
  const keys = ['input', 'cond', 'n', 'variant', 'compare', 'cmp', 'carton', 'mass', 'crack', 'hard', 'mu', 'noise', 'src', 'seed'];
  const h = new URLSearchParams();
  for (const k of keys) h.set(k, typeof S[k] === 'boolean' ? (S[k] ? '1' : '0') : S[k]);
  history.replaceState(null, '', `#${h}`);
}

// ---------------------------------------------------------------- loading
let mj, index, tex, camSpec;
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
  camSpec = await (await fetch('assets/camera.json')).json();
  tex = makeTextures({ wood: 'assets/wood.jpg', pulp: 'assets/pulp.jpg' });
  progress('Loading the policies…', 0.9);
  initThree();
  buildPanel();
  await restart();
  $('loading').classList.add('done');
  requestAnimationFrame(frame);
}

const policyName = (cond, n, variant, input = S.input) => (input === 'pixels' ? `px_${cond}_n${n}`
  : variant && cond === 'C' && n === 25 ? `var_${variant}` : `${cond}_n${n}`);

// ---------------------------------------------------------------- one side of the view
// The simulation runs in a Web Worker (app/worker.js -> sim/run.js). This side keeps what it sends back: a record
// per control step (state, squeeze, contacts, camera images) to draw, chart and scrub through.
class Session {
  constructor(side, model) {
    this.side = side;
    this.m = model;
    this.view = new SceneView(model, tex);
    this.view.scene.environment = envMap;
    this.replay = new mj.MjData(model);
    this.cams = new Cameras(mj, model, camSpec);
    this.pip = makePip(side);
    this.gripper = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'gripper');
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => (e.data.type === 'ready' ? resolve() : reject(new Error(e.data.message)));
    });
    this.worker.postMessage({ type: 'init', base: new URL('.', location.href).href });
    this.frames = []; this.outcomes = []; this.pose = null; this.runId = 0; this.inflight = false;
    this.pending = 0; this.isDone = true;
  }

  async start(policyName, label, color, kinds, seed) {
    await this.ready;
    const id = ++this.runId;
    this.label = label; this.color = color; this.kinds = kinds;
    this.frames = []; this.outcomes = []; this.pose = null; this.isDone = false; this.pending = 0; this.simulated = 0;
    this.inflight = false;           // the old run's reply is dropped below; the worker answers 'start' before any new 'run'
    this._pipShown = null; this._pipFrame = null; this.lastView = null;
    this.worker.onmessage = (e) => this.onMessage(e.data);
    const opts = { noise: S.noise, mu: S.mu, src: S.src, saliency: S.saliency, contacts: S.contacts };
    this.worker.postMessage({ type: 'start', id, policy: policyName, kinds, seed, opts });
    await new Promise((resolve) => { this._started = resolve; });
  }

  onMessage(msg) {
    if (msg.type === 'error') { console.error('worker:', msg.message); return; }
    if (msg.id !== this.runId) return;                 // a reply for a run that was replaced
    if (msg.type === 'started') {
      this.pixels = msg.pixels;
      this.add(msg.rec);
      this._started();
      return;
    }
    if (msg.type === 'ran') {
      this.inflight = false;
      this.simulated += msg.n;
      for (const r of msg.recs) {
        this.add(r);
        if (r.event) { this.outcomes.push(r.event); toast(this, r.event.out); renderScores(); }
      }
      this.pose = msg.pose;
      if (msg.done) { this.isDone = true; renderScores(); }
    }
  }

  add(r) {
    if (r.view) this.lastView = r.view;
    r.sal = r.view && r.view.maps ? r.view.maps : null;
    delete r.view;
    this.frames.push(r);
  }

  get done() { return this.isDone; }

  // ask the worker for the physics steps owed (at most ~0.3 s of sim per request)
  pump(steps) {
    if (this.done) return;
    this.pending = Math.min(this.pending + steps, 150);
    if (this.inflight || this.pending < 1) return;
    const n = Math.floor(this.pending);
    this.pending -= n;
    this.inflight = true;
    this.worker.postMessage({ type: 'run', id: this.runId, n });
  }

  grabAt(body, target) { this.worker.postMessage({ type: 'grab', body, target }); }
  release() { this.worker.postMessage({ type: 'release' }); }

  // pose the view from the newest state, or from a recorded frame while scrubbing
  show(f = null) {
    const fr = this.frames[f === null ? this.frames.length - 1 : Math.min(f, this.frames.length - 1)];
    if (!fr) return;
    this.replay.qpos.set(f === null && this.pose ? this.pose : fr.qpos);
    mj.mj_kinematics(this.m, this.replay);
    this.view.update(this.replay, { sensors: S.sensors, padForces: fr.pads });
    this.showPip(f);
    this.view.setContacts(S.contacts ? fr.contacts : []);
    const gb = grab && grab.session === this && f === null ? grab : null;
    this.view.setSpring(gb ? Array.from(this.replay.xpos.subarray(3 * gb.body, 3 * gb.body + 3)) : null, gb ? gb.target : null);
  }

  ee() {
    const d = this.replay, b = this.gripper, R = d.xmat.subarray(9 * b), p = d.xpos.subarray(3 * b);
    const L = [0.0117, -0.0002, -0.0830];               // env.js PINCH_LOCAL
    return [0, 1, 2].map((i) => p[i] + R[3 * i] * L[0] + R[3 * i + 1] * L[1] + R[3 * i + 2] * L[2]);
  }

  // the policy's two camera images, with where it looks
  showPip(f) {
    const el = this.pip;
    el.hidden = !this.pixels || this.side >= active().length;
    if (el.hidden) return;
    let v;
    if (f === null) v = this.lastView;
    else {
      // scrubbing: re-render the cameras from the recorded state; saliency is kept per step
      if (this._pipFrame === f) return;
      this._pipFrame = f;
      const fr = this.frames[Math.min(f, this.frames.length - 1)];
      this.replay.qpos.set(fr.qpos);
      v = { frames: this.cams.render(this.replay), maps: fr.sal };
      mj.mj_kinematics(this.m, this.replay);
    }
    if (!v || v === this._pipShown) return;
    this._pipShown = v;
    drawPip(el, v);
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

// ---------------------------------------------------------------- picture-in-picture: what the pixel policy sees
// The overlay is gradient saliency: how much each pixel moves the policy's next action (sim/policy.js saliency()).
function makePip(side) {
  const el = document.createElement('div');
  el.className = 'pip'; el.hidden = true;
  el.innerHTML = ['Workspace camera', 'Wrist camera'].map((t) =>
    `<figure><canvas width="128" height="128"></canvas><figcaption>${t}</figcaption></figure>`).join('');
  el.dataset.side = side;
  return el;
}

const heatCanvas = Object.assign(document.createElement('canvas'), { width: 64, height: 64 });
function drawPip(el, v) {
  el.querySelectorAll('canvas').forEach((cv, c) => {
    const g = cv.getContext('2d'), R = camSpec.res, im = g.createImageData(R, R), src = v.frames[c];
    for (let i = 0; i < R * R; i++) { im.data[4 * i] = src[3 * i]; im.data[4 * i + 1] = src[3 * i + 1]; im.data[4 * i + 2] = src[3 * i + 2]; im.data[4 * i + 3] = 255; }
    const tmp = drawPip.tmp || (drawPip.tmp = Object.assign(document.createElement('canvas'), { width: R, height: R }));
    tmp.getContext('2d').putImageData(im, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(tmp, 0, 0, cv.width, cv.height);
    if (!S.saliency || !v.maps) return;
    // 3x3 box blur for display, then scale to the 99th percentile so one hot pixel doesn't wash out the rest
    const m = v.maps[c], b = new Float32Array(R * R);
    for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
      let t = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < R && xx >= 0 && xx < R) { t += m[yy * R + xx]; n++; }
      }
      b[y * R + x] = t / n;
    }
    const top = Float32Array.from(b).sort()[Math.floor(0.99 * (R * R - 1))] || 1;
    const hc = heatCanvas.getContext('2d'), h = hc.createImageData(R, R);
    for (let i = 0; i < R * R; i++) {
      const a = Math.min(1, b[i] / top);
      h.data[4 * i] = 255; h.data[4 * i + 1] = Math.round(40 + 190 * a); h.data[4 * i + 2] = 20; h.data[4 * i + 3] = Math.round(230 * a * a);
    }
    hc.putImageData(h, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(heatCanvas, 0, 0, cv.width, cv.height);
  });
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
  for (const s of sessions) $('pips').appendChild(s.pip);
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
  s.grabAt(grab.body, grab.target);
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
    grab.session.grabAt(grab.body, t);
  }
}
function onUp() {
  if (!grab) return;
  grab.session.release();
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
  const names = [policyName(S.cond, S.n, S.variant)];
  if (S.compare) names.push(policyName(S.cmp, S.n, ''));
  const kinds = cartonKinds(new Rng(S.seed * 7919 + 13));
  const labels = [policyLabel(S.cond, S.variant), policyLabel(S.cmp, '')];
  const colors = [css(`--c-${S.cond}`), css(`--c-${S.cmp}`)];
  if (grab) onUp();
  await Promise.all(names.map((nm, i) => sessions[i].start(nm, `${labels[i]} · ${S.n} robot demos`, colors[i], kinds, S.seed)));
  if (id !== runId) return;
  for (const s of sessions) s.view.showCollision(S.collision);
  $('scrub').value = 0;
  simWin = [];
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

// real-time factor actually achieved, shown when the simulation can't keep up with the requested speed
let simWin = [];
function trackSpeed(dt) {
  const act = active(), sim = Math.min(...act.map((s) => s.simulated || 0)) / 500;
  simWin.push([performance.now(), sim]);
  while (simWin.length > 2 && performance.now() - simWin[0][0] > 2000) simWin.shift();
  const [t0, s0] = simWin[0], [t1, s1] = simWin[simWin.length - 1];
  const rate = t1 - t0 > 500 ? (s1 - s0) / ((t1 - t0) / 1000) : null;
  const el = $('rate');
  el.hidden = !(rate !== null && rate < 0.85 * S.speed && !act.every((s) => s.done));
  if (!el.hidden) el.textContent = `running at ${rate.toFixed(2)}× (your CPU is the limit)`;
}

function frame(now) {
  requestAnimationFrame(frame);                  // first, so one bad frame can't stop the loop
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (running && !scrubbing()) {
    for (const s of active()) s.pump(dt * 500 * S.speed);
    trackSpeed(dt);
    if (active().every((s) => s.done)) { setRunning(false); renderScores(); }
    const len = Math.max(...active().map((s) => s.frames.length)) - 1;
    $('scrub').max = len; $('scrub').value = len;
  }
  draw();
}

function draw() {
  const { W, H, rects, vertical } = layout();
  const dpr = renderer.getPixelRatio();
  if (renderer.domElement.width !== Math.round(W * dpr) || renderer.domElement.height !== Math.round(H * dpr)) renderer.setSize(W, H, false);
  const act = active();
  // camera: tween to presets, or follow the pinch point
  if (camMode === 'follow') {
    const ee = act[0].ee();
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
  // labels and the chart; a session that is restarting has no records yet
  if (act.some((s) => !s.frames.length)) return;
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
function updateLabels(rebuild, rects = layout().rects, vertical = layout().vertical) {
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
  sessions.forEach((s, i) => { if (i >= act.length) s.pip.hidden = true; });
  act.forEach((s, i) => {
    const r = rects[i], el = labelEls[i];
    el.style.left = `${r.x + 12}px`; el.style.top = `${r.y + 12}px`;
    const ph = s.pip.offsetHeight || 0;
    s.pip.style.left = `${r.x + 12}px`;
    s.pip.style.top = `${Math.max(r.y + 56, r.y + r.h - ph - (i === act.length - 1 || !vertical ? 52 : 12))}px`;
    const fr = scrubbing() ? s.frames[Math.min(viewFrame, s.frames.length - 1)] : s.frames[s.frames.length - 1];
    const live = fr.grip;
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
function successOf(cond, n, input = S.input) {
  const p = index.find((q) => q.name === policyName(cond, n, '', input));
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
  document.querySelectorAll('#input button').forEach((b) => { b.onclick = () => { S.input = b.dataset.input; syncPanel(); restart(); }; });
  $('v-saliency').onchange = (e) => {
    S.saliency = e.target.checked;
    for (const s of sessions) { s._pipShown = null; s.worker.postMessage({ type: 'opts', opts: { saliency: S.saliency } }); }
  };
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
  $('v-contacts').onchange = (e) => { S.contacts = e.target.checked; for (const s of sessions) s.worker.postMessage({ type: 'opts', opts: { contacts: S.contacts } }); };
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
  document.querySelectorAll('#input button').forEach((b) => b.classList.toggle('on', b.dataset.input === S.input));
  $('input-note').innerHTML = S.input === 'pixels'
    ? 'Two 64&times;64 cameras (picture-in-picture, bottom left) plus the arm\'s own joint state and touch. It has to find the egg in the image.'
    : 'The egg\'s position, measured to 3 mm, plus the arm\'s joint state and touch. The original study.';
  $('v-saliency').disabled = S.input !== 'pixels';
  const vOk = S.cond === 'C' && S.n === 25 && S.input === 'state';
  $('variant').disabled = !vOk;
  if (!vOk) S.variant = '';
  $('variant').value = S.variant;
  $('variant').title = vOk ? '' : 'Recipe variants exist for the touch policy with 25 robot demos, seeing egg position';
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
  const nSeeds = meta.success_seeds.length;
  $('gain-cap').innerHTML = (S.input === 'pixels' ? 'Solid: sees camera pixels. Dashed: sees egg position. ' : '') +
    `Python evaluation, ${nSeeds} seeds × 60 eggs each. Loaded: seed ${meta.seed}, <b>${Math.round(100 * meta.success)}%</b>` +
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
  if (S.input === 'pixels') {
    for (const c of 'ABCD') {
      const pts = ns.map((n) => [x(n), y(successOf(c, n, 'state'))]);
      s += `<polyline points="${pts.map((p) => p.join(',')).join(' ')}" fill="none" stroke="var(--c-${c})" stroke-width="1.25" stroke-dasharray="3 3" opacity="0.45"/>`;
    }
  }
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
