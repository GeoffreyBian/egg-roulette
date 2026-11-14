"""Collect demonstrations. Every attempt is kept with per-step labels; training code filters as needed.

Per episode: obs (T, 20) with touch as the last 3 columns, act (T, 4), phase (T,), slip (T,), kind, outcome.

  teleop         robot teleoperator, can't feel: 100 successes plus the failures on the way
  human          glove wearer, grip strategy from real human grasps: 300 successes
  human_heavy    heavy-handed glove wearers (start firm, big margin): 120 successes
  human_diverse  glove wearer on a spread of egg weights and fragilities: 150 successes
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from .scene import ROOT

DATA = ROOT / "data"
PHASES = ["approach", "descend", "close", "preload", "lift", "carry", "lower", "open", "retreat"]
MAX_STEPS = 330
SETS = {  # name: (demonstrator, successes per worker, workers, seed base)
    "teleop": ("teleop", 10, 10, 400),
    "human": ("human", 30, 10, 300),
    "human_heavy": ("heavy", 12, 10, 700),
    "human_diverse": ("diverse", 15, 10, 800),
}


def _worker(args):
    who, n_success, seed = args
    from .demos import HumanGloveDemo, TeleopDemo, diverse_kind, run_episode
    from .env import EggEnv, flat_obs, setup_episode
    env = EggEnv(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    eps, ok = [], 0
    while ok < n_success:
        kind = diverse_kind(rng) if who == "diverse" else ("raw" if ok % 2 == 0 else "stone")
        setup_episode(env, rng, kind)
        demo = TeleopDemo(env, rng) if who == "teleop" else HumanGloveDemo(env, rng, heavy_handed=who == "heavy")
        O, A, P, S = [], [], [], []

        def rec(o, a, info):
            O.append(flat_obs(o)); A.append(np.asarray(a, np.float32))
            P.append(PHASES.index(demo.phase)); S.append(int(info.slip > 0.02 and info.grip_n > 0.05))
        _, out = run_episode(env, demo, max_steps=MAX_STEPS, record=rec)
        ok += out == "success"
        eps.append(dict(obs=np.array(O), act=np.array(A), phase=np.array(P, np.int8), slip=np.array(S, np.int8),
                        kind=kind, outcome=out))
    return eps


def collect(names=None):
    DATA.mkdir(exist_ok=True)
    for name in names or SETS:
        who, n, workers, base = SETS[name]
        t0 = time.time()
        with mp.Pool(min(workers, mp.cpu_count())) as pool:
            res = pool.map(_worker, [(who, n, base + i) for i in range(workers)])
        eps = [e for r in res for e in r]
        np.savez_compressed(DATA / f"demos_{name}.npz", eps=np.array(eps, dtype=object))
        outs = {o: sum(e["outcome"] == o for e in eps) for o in sorted({e["outcome"] for e in eps})}
        print(f"{name:14s} {len(eps):4d} attempts, {100 * outs.get('success', 0) / len(eps):.0f}% success, "
              f"{outs}  ({time.time() - t0:.0f}s)", flush=True)


def load(name):
    return list(np.load(DATA / f"demos_{name}.npz", allow_pickle=True)["eps"])
