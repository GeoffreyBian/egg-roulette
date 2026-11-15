// Pixel policies: checks the browser port against Python (web/test/reference_pixels.json).
//   1. cameras: rendering the recorded sim states with sim/camera.js gives Python's frames
//   2. policy: the JS CNN, fed Python's frames and inputs, gives Python's actions
//   3. closed loop: success of the whole JS stack (physics -> cameras -> CNN) on fresh eggs vs the Python sweep
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import load from '@mujoco/mujoco';
import { EggEnv } from '../sim/env.js';
import { Cameras } from '../sim/camera.js';
import { PixelPolicy, PixelRunner } from '../sim/policy.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const assets = path.join(here, '../assets');
const mj = await load();
mj.FS.mkdir('/w'); mj.FS.mkdir('/w/meshes');
for (const f of fs.readdirSync(path.join(assets, 'meshes'))) mj.FS.writeFile(`/w/meshes/${f}`, fs.readFileSync(path.join(assets, 'meshes', f)));
mj.FS.writeFile('/w/scene.xml', fs.readFileSync(path.join(assets, 'scene.xml'), 'utf8'));
const model = mj.MjModel.from_xml_path('/w/scene.xml');
const spec = JSON.parse(fs.readFileSync(path.join(assets, 'camera.json')));
const index = JSON.parse(fs.readFileSync(path.join(assets, 'policies/index.json')));
const pols = {};
for (const meta of index.filter((p) => p.input === 'pixels')) {
  pols[meta.name] = new PixelPolicy(meta, fs.readFileSync(path.join(assets, `policies/${meta.name}.bin`)).buffer.slice(0));
}
const ref = JSON.parse(fs.readFileSync(path.join(here, 'reference_pixels.json')));
const quick = process.argv.includes('--quick');
// frames are base64 uint8, cameras x 64 x 64 x 3: split into one image per camera
const unpack = (b64) => { const u = new Uint8Array(Buffer.from(b64, 'base64')), n = u.length / 2; return [u.slice(0, n), u.slice(n)]; };
let failed = false;
const check = (ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); if (!ok) failed = true; };

// ---------------------------------------------------------------- 1. cameras
const env = new EggEnv(mj, model, { seed: 1 });
const cams = new Cameras(mj, model, spec);
let nPix = 0, nDiff = 0, worst = 0, nFrames = 0;
for (const ep of ref) {
  env.configure(ep.init.kinds, ep.init.egg_r);
  ep.qpos.forEach((q, t) => {
    env.d.qpos.set(q);
    mj.mj_kinematics(model, env.d);
    const ims = cams.render(env.d);
    ims.forEach((im, c) => {
      const want = unpack(ep.frames[t])[c];
      for (let i = 0; i < im.length; i++) {
        const dv = Math.abs(im[i] - want[i]);
        nPix++; if (dv) nDiff++;
        worst = Math.max(worst, dv);
      }
    });
    nFrames++;
  });
}
check(nDiff / nPix < 1e-3, `cameras: ${nFrames} states x 2 cameras, ${nDiff} of ${nPix} channel values differ ` +
  `(${(100 * nDiff / nPix).toFixed(4)}%), worst by ${worst}/255`);

// ---------------------------------------------------------------- 2. policy
let maxErr = 0, nSteps = 0, salErr = 0, nSal = 0;
for (const ep of ref) {
  const runner = new PixelRunner(pols[ep.policy], null);
  ep.frames.forEach((fr, t) => {
    const a = runner.act(ep.obs[t], null, unpack(fr));
    for (let j = 0; j < 4; j++) maxErr = Math.max(maxErr, Math.abs(a[j] - ep.actions[t][j]));
    nSteps++;
    if (ep.saliency[t]) {
      // gradient saliency vs torch autograd, relative to the map's largest value
      const js = runner.pol.saliency();
      ep.saliency[t].forEach((cam, c) => {
        const want = cam.flat(), top = Math.max(...want);
        for (let i = 0; i < want.length; i++) salErr = Math.max(salErr, Math.abs(js[c][i] - want[i]) / top);
      });
      nSal++;
    }
  });
}
check(maxErr < 1e-4, `policy: max |action - python| over ${nSteps} steps = ${maxErr.toExponential(2)}`);
check(salErr < 1e-3, `saliency: ${nSal} frames x 2 cameras vs torch autograd, max error ${salErr.toExponential(2)} of the peak`);

// ---------------------------------------------------------------- 3. closed loop
const gain = JSON.parse(fs.readFileSync(path.join(here, '../../results/gain_pixels.json')));
const N = quick ? 10 : 60;
for (const name of ['px_C_n25', 'px_A_n25']) {
  const meta = pols[name].meta;
  const e = new EggEnv(mj, model, { seed: 4243 });
  const runner = new PixelRunner(pols[name], new Cameras(mj, model, spec));
  let ok = 0;
  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    e.setupEpisode(i % 2 ? 'stone' : 'raw');
    runner.reset();
    let obs = e.obs(), out = 'timeout', basket = 0;
    for (let t = 0; t < 330; t++) {
      const a = runner.act(obs, e.d);
      e.step(a);
      obs = e.obs();
      const st = e.eggStatus();
      if (st === 'cracked' || st === 'dropped') { out = st; break; }
      if (st === 'basket' && a[3] < 0 && ++basket > 8) { out = 'success'; break; }
    }
    ok += out === 'success';
  }
  const py = gain.find((r) => r.cond === meta.cond && r.n_robot === meta.n_robot && r.seed === meta.seed);
  const pj = ok / N, se = Math.sqrt(pj * (1 - pj) / N + py.success * (1 - py.success) / py.n);
  const z = Math.abs(pj - py.success) / Math.max(se, 1e-9);
  check(quick || z < 2.6, `closed loop ${name}: js ${(100 * pj).toFixed(0)}% on ${N} eggs, python ${(100 * py.success).toFixed(0)}% ` +
    `on ${py.n} (z = ${z.toFixed(2)}), ${((performance.now() - t0) / N / 1000).toFixed(2)} s/egg`);
}
process.exit(failed ? 1 : 0);
