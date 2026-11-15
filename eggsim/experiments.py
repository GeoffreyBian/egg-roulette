"""Every experiment in the README, as resumable job lists. Results land in results/*.json.

  gain        4 conditions x {10, 25, 50, 100} robot demos x 5 seeds
  ablations   what else limits the touch-trained policy (labels, timing, data volume, sensing, demonstrators)
  robustness  the n=25 policies of A, C, D on a grid of pad friction x egg mass they never trained on
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np

from .scene import ROOT

RESULTS = ROOT / "results"
POLICIES = ROOT / "data/policies"
SEEDS = [0, 1, 2, 3, 4]
N_ROBOT_GRID = [10, 25, 50, 100]

ABLATIONS = {
    "C base": {},
    "C memory 1 step": dict(hist=1),
    "C memory 6 steps": dict(hist=6),
    "C no action chunks": dict(chunk=1),
    "C chunks of 16": dict(chunk=16),
    "C 50 human demos": dict(n_human=50),
    "C 300 human demos": dict(n_human=300),
    "C + phase labels": dict(aux_phase=True),
    "C + object labels": dict(aux_kind=True),
    "C + slip labels": dict(aux_slip=True),
    "C + all three labels": dict(aux_phase=True, aux_kind=True, aux_slip=True),
    "C + failed demos, labelled": dict(success_cond=True),
    "C + failed demos, unlabelled": dict(unfiltered=True),
    "C touch lags video 100 ms": dict(touch_lag=3),
    "C touch lags video 300 ms": dict(touch_lag=9),
    "C glove off by up to 30%": dict(miscal=0.3),
    "C best 150 of 300 demos": dict(quality_top=True),
    "C grip pressure only": dict(touch_mask=(1, 0, 0)),
    "C load (shear) only": dict(touch_mask=(0, 1, 0)),
    "C slip only": dict(touch_mask=(0, 0, 1)),
    "C no touch sensor on robot": dict(touch=False),
    "C servo-current estimate": dict(proxy=True),
    "C glove-trained, run on servo est.": dict(test_proxy=True),
    "C glove at 10 Hz": dict(touch_rate=3),
    "C glove at 5 Hz": dict(touch_rate=6),
    "C 30% heavy-handed demonstrators": dict(heavy_frac=0.3),
    "C 60% heavy-handed demonstrators": dict(heavy_frac=0.6),
    "C 60% heavy-handed, labelled": dict(heavy_frac=0.6, heavy_label=True),
    "C human demos on varied eggs": dict(human_src="human_diverse"),
    "A base": dict(touch=False, human=False),
    "A + object labels": dict(touch=False, human=False, aux_kind=True),
    "A memory 6 steps": dict(touch=False, human=False, hist=6),
}
# these also run the unseen-egg suites
GEN = {"C base", "A base", "C no touch sensor on robot", "C servo-current estimate", "C human demos on varied eggs"}

MU_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
MASS_GRID = [0.6, 0.8, 1.0, 1.2, 1.4]
ROBUST_EGGS = 20


# ---------------------------------------------------------------- jobs
def _summary(res, steps):
    n = sum(sum(v.values()) for v in res.values())
    k = sum(v.get("success", 0) for v in res.values())
    return dict(success=k / n, n=n, detail=res, mean_steps_success=float(np.mean(steps)) if steps else None)


def _gain_job(args):
    cond, n_robot, seed = args
    from .policy import CONDITIONS, config, evaluate, save, train
    t0 = time.time()
    pol = train(config(**CONDITIONS[cond], n_robot=n_robot), seed)
    save(pol, POLICIES / f"{cond}_n{n_robot}_s{seed}.pt")
    out = dict(cond=cond, n_robot=n_robot, seed=seed, **_summary(*evaluate(pol, seed)))
    print(f"gain {cond} n={n_robot:3d} s={seed}  {100 * out['success']:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    return out


def _ablation_job(args):
    name, seed = args
    from .policy import GEN_SUITES, config, evaluate, train
    t0 = time.time()
    pol = train(config(**ABLATIONS[name]), seed)
    out = dict(name=name, seed=seed, **_summary(*evaluate(pol, seed)))
    if name in GEN:
        out["gen"] = {k: evaluate(pol, 100 + seed, n=30, kinds=(k,))[0][k] for k in GEN_SUITES}
    print(f"ablation {name:36s} s={seed}  {100 * out['success']:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    return out


def _robust_job(args):
    cond, seed, mu, ms = args
    from .policy import evaluate, load_policy
    pol = load_policy(POLICIES / f"{cond}_n25_s{seed}.pt")
    res, steps = evaluate(pol, 200 + seed, n=ROBUST_EGGS, env_kw=dict(mu=mu, mass_scale=ms))
    return dict(cond=cond, seed=seed, mu=mu, mass_scale=ms, **_summary(res, steps))


# ---------------------------------------------------------------- runner
def _run(fn, jobs, path, key, workers=None):
    """Runs jobs not already in results/<path>, saving after each one so a sweep can be resumed."""
    RESULTS.mkdir(exist_ok=True)
    POLICIES.mkdir(parents=True, exist_ok=True)
    f = RESULTS / path
    done = json.loads(f.read_text()) if f.exists() else []
    have = {key(r) for r in done}
    todo = [j for j in jobs if tuple(j) not in have]
    print(f"{path}: {len(todo)} of {len(jobs)} jobs to run", flush=True)
    t0 = time.time()
    with mp.Pool(workers or mp.cpu_count()) as pool:
        for r in pool.imap_unordered(fn, todo):
            done.append(r)
            f.write_text(json.dumps(done, indent=1))
    print(f"{path}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


def gain():
    jobs = [(c, n, s) for n in N_ROBOT_GRID for c in "ABCD" for s in SEEDS]
    _run(_gain_job, jobs, "gain.json", lambda r: (r["cond"], r["n_robot"], r["seed"]))


def ablations():
    jobs = [(n, s) for n in ABLATIONS for s in SEEDS]
    _run(_ablation_job, jobs, "ablations.json", lambda r: (r["name"], r["seed"]))


def robustness():
    jobs = [(c, s, mu, ms) for c in "ACD" for s in SEEDS for mu in MU_GRID for ms in MASS_GRID]
    _run(_robust_job, jobs, "robustness.json", lambda r: (r["cond"], r["seed"], r["mu"], r["mass_scale"]))
