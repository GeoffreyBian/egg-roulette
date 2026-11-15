// Checks the browser port against Python (web/test/reference.json, written by `python -m eggsim web-export`).
//   1. policy: the JS MLP + ensembling, fed the exact inputs Python fed, gives the same actions
//   2. physics: replaying Python's actions open-loop through env.js tracks Python's squeeze force and pinch
//      point. Contact-rich physics is chaotic, so exact agreement is only expected until the first
//      floating-point difference (WebAssembly vs native libm) has had time to grow.
//   3. closed loop: success rates of the JS stack on fresh eggs, against the Python sweep for the same policy.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import load from '@mujoco/mujoco';
import { EggEnv } from '../sim/env.js';
import { Policy, Runner } from '../sim/policy.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const assets = path.join(here, '../assets');
const mj = await load();
mj.FS.mkdir('/w'); mj.FS.mkdir('/w/meshes');
for (const f of fs.readdirSync(path.join(assets, 'meshes'))) mj.FS.writeFile(`/w/meshes/${f}`, fs.readFileSync(path.join(assets, 'meshes', f)));
mj.FS.writeFile('/w/scene.xml', fs.readFileSync(path.join(assets, 'scene.xml'), 'utf8'));
const model = mj.MjModel.from_xml_path('/w/scene.xml');
const index = JSON.parse(fs.readFileSync(path.join(assets, 'policies/index.json')));
const policies = {};
for (const meta of index) policies[meta.name] = new Policy(meta, fs.readFileSync(path.join(assets, `policies/${meta.name}.bin`)).buffer);
const ref = JSON.parse(fs.readFileSync(path.join(here, 'reference.json')));
const quick = process.argv.includes('--quick');

// ---------------------------------------------------------------- 1. policy parity
let worst = 0;
for (const ep of ref) {
  const runner = new Runner(policies[ep.policy]);
  ep.obs.forEach((o, t) => {
    const a = runner.act(o);
    for (let j = 0; j < 4; j++) worst = Math.max(worst, Math.abs(a[j] - ep.actions[t][j]));
  });
}
console.log(`policy: max |action - python| over ${ref.reduce((s, e) => s + e.obs.length, 0)} steps = ${worst.toExponential(2)}`);

// ---------------------------------------------------------------- 2. physics replay
const env = new EggEnv(mj, model, { seed: 1 });
for (const ep of ref) {
  const I = ep.init;
  env.configure(I.kinds, I.egg_r);
  mj.mj_resetData(model, env.d);
  env.d.qpos.set(I.qpos); env.d.qvel.set(I.qvel); env.d.ctrl.set(I.ctrl); env.d.qacc_warmstart.set(I.warmstart);
  mj.mj_forward(model, env.d);
  env.target = I.target; env.drop = I.drop; env.eeTarget = [...I.ee_target];
  env.prevAction = [0, 0, 0, 0]; env.last = { grip: 0, shear: 0, slip: 0 };
  let diverge = null, outcome = 'timeout', basket = 0, maxGripErr = 0;
  for (let t = 0; t < ep.actions.length; t++) {
    const info = env.step(ep.actions[t]);
    const ee = env.ee(), dee = Math.hypot(...ee.map((v, i) => v - ep.ee[t][i]));
    if (diverge === null && (dee > 0.002 || Math.abs(info.grip - ep.grip[t]) > 0.1)) diverge = t;
    if (diverge === null) maxGripErr = Math.max(maxGripErr, Math.abs(info.grip - ep.grip[t]));
    const st = env.eggStatus();
    if (st === 'cracked' || st === 'dropped') { outcome = st; break; }
    if (st === 'basket' && ep.actions[t][3] < 0 && ++basket > 8) { outcome = 'success'; break; }
  }
  console.log(`replay ${ep.policy} ${ep.kind.padEnd(5)} python=${ep.outcome.padEnd(8)} js=${outcome.padEnd(8)} ` +
    `tracks for ${diverge ?? ep.actions.length}/${ep.actions.length} steps (grip err ${maxGripErr.toFixed(4)} N)`);
}

// ---------------------------------------------------------------- 3. closed-loop success
const gain = JSON.parse(fs.readFileSync(path.join(here, '../../results/gain.json')));
const N = quick ? 10 : 60;
for (const name of ['C_n25', 'A_n25']) {
  const meta = policies[name].meta;
  const e = new EggEnv(mj, model, { seed: 4242 });
  const runner = new Runner(policies[name]);
  const counts = {};
  let ok = 0;
  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    const kind = i % 2 ? 'stone' : 'raw';
    e.setupEpisode(kind);
    runner.reset();
    let obs = e.obs(), out = 'timeout', basket = 0;
    for (let t = 0; t < 330; t++) {
      const a = runner.act(obs);
      e.step(a);
      obs = e.obs();
      const st = e.eggStatus();
      if (st === 'cracked' || st === 'dropped') { out = st; break; }
      if (st === 'basket' && a[3] < 0 && ++basket > 8) { out = 'success'; break; }
    }
    counts[`${kind}:${out}`] = (counts[`${kind}:${out}`] || 0) + 1;
    ok += out === 'success';
  }
  const py = gain.find((r) => r.cond === meta.cond && r.n_robot === meta.n_robot && r.seed === meta.seed);
  const se = Math.sqrt(py.success * (1 - py.success) / N);
  console.log(`closed loop ${name} (seed ${meta.seed}): js ${(100 * ok / N).toFixed(0)}% on ${N} eggs, python ` +
    `${(100 * py.success).toFixed(0)}% on ${py.n}; ±${(196 * se).toFixed(0)} pts at 95%  ${JSON.stringify(counts)}  ` +
    `${((performance.now() - t0) / N / 1000).toFixed(2)} s/egg`);
}
