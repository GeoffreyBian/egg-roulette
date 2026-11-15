// One simulation per worker: MuJoCo, the policy and the carton logic (sim/run.js). The page only draws. Two
// workers run the side-by-side view on two cores.
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.17.0/mujoco.js';
import { CartonRun } from '../sim/run.js';
import { PixelPolicy, Policy } from '../sim/policy.js';

let mj, run, index;
const policies = new Map();

async function init({ base }) {
  mj = await loadMujoco();
  const get = async (p) => { const r = await fetch(new URL(p, base)); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r; };
  const xml = await (await get('assets/scene.xml')).text();
  const files = [...new Set([...xml.matchAll(/<mesh [^>]*file="([^"]+)"/g)].map((m) => m[1]))];
  mj.FS.mkdir('/w'); mj.FS.mkdir('/w/meshes');
  await Promise.all(files.map(async (f) => mj.FS.writeFile(`/w/meshes/${f}`, new Uint8Array(await (await get(`assets/meshes/${f}`)).arrayBuffer()))));
  mj.FS.writeFile('/w/scene.xml', xml);
  const model = mj.MjModel.from_xml_path('/w/scene.xml');
  index = await (await get('assets/policies/index.json')).json();
  const camSpec = await (await get('assets/camera.json')).json();
  run = new CartonRun(mj, model, camSpec);
  init.get = get;
}

async function policy(name) {
  if (!policies.has(name)) {
    const meta = index.find((p) => p.name === name);
    const Kind = meta.input === 'pixels' ? PixelPolicy : Policy;
    policies.set(name, new Kind(meta, await (await init.get(`assets/policies/${name}.bin`)).arrayBuffer()));
  }
  return policies.get(name);
}

// records carry Float64Arrays and image buffers: transfer them instead of copying
function transferables(recs) {
  const t = [];
  for (const r of recs) {
    t.push(r.qpos.buffer);
    if (r.view) for (const f of r.view.frames) t.push(f.buffer);
  }
  return t;
}

let busy = Promise.resolve();
self.onmessage = (e) => {
  const msg = e.data;
  // messages are handled strictly in order
  busy = busy.then(() => handle(msg)).catch((err) => self.postMessage({ type: 'error', message: String(err && err.message || err) }));
};

async function handle(msg) {
  if (msg.type === 'init') { await init(msg); self.postMessage({ type: 'ready' }); return; }
  if (msg.type === 'start') {
    const pol = await policy(msg.policy);
    const rec = run.start(pol, msg.kinds, msg.seed, msg.opts);
    self.postMessage({ type: 'started', id: msg.id, rec, pixels: run.pixels }, transferables([rec]));
    return;
  }
  if (msg.type === 'run') {
    // advance up to n physics steps; the final state goes back as the pose to draw
    const recs = [], t0 = performance.now();
    let n = 0;
    while (n < msg.n && !run.done) {
      const rec = run.tick();
      n++;
      if (rec) recs.push(rec);
      if (performance.now() - t0 > 40) break;               // stay responsive: report at least every 40 ms
    }
    const pose = Float64Array.from(run.env.d.qpos);
    self.postMessage({ type: 'ran', id: msg.id, n, recs, pose, done: run.done }, transferables(recs).concat([pose.buffer]));
    return;
  }
  if (msg.type === 'grab') { run.grab = { body: msg.body, target: msg.target }; return; }
  if (msg.type === 'release') { run.release(); return; }
  if (msg.type === 'opts') { Object.assign(run.opts, msg.opts); }
}
