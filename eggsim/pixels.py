"""Pixel policies: the same task and data as policy.py, but the policy sees two 64x64 camera images instead of the
egg's position. It still gets its own joint state, the drop spot it was told to use, and (in C and D) touch.

  render       demos' sim states -> data/pixels_<name>.npy (frames, 2 cameras, 64, 64, 3) uint8
  gain         4 conditions x {10, 25, 50, 100} robot demos x seeds -> results/gain_pixels.json

Encoder: per camera, three convolutions and a spatial softmax (32 keypoints, Levine et al. 2016), so the policy's
"where am I looking" is an explicit 16x16 attention map per keypoint; the browser draws these as the saliency
overlay. Trunk and action head are policy.py's: an MLP predicting chunks of 8 actions, temporal ensembling online.
Training augments with random shifts of up to 4 pixels (DrQ).

Assumption to keep in mind when reading results: human demos are rendered with the robot's gripper where the
hand would be, i.e. perfect visual retargeting from hand to gripper. Video of a real hand would not look like this.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np

from .camera import RES, Cameras
from .collect import DATA, load
from .policy import BASE, CONDITIONS, config
from .scene import ROOT

PROPRIO = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16]   # ee, jaw angle, drop spot - ee, previous action
TOUCH = [17, 18, 19]
KEYPOINTS = 32
SHIFT = 4
STEPS = 8000
BATCH = 128
RESULTS = ROOT / "results"
POLICIES = ROOT / "data/policies_pixels"
SEEDS = [0, 1, 2]
N_ROBOT_GRID = [10, 25, 50, 100]
EVAL_EPISODES = 60


# ---------------------------------------------------------------- rendering the demos
def _render_job(args):
    name, lo, hi = args
    from .env import EggEnv
    env = EggEnv(seed=0)
    cams = Cameras(env.m)
    st = list(np.load(DATA / f"states_{name}.npz", allow_pickle=True)["eps"])
    eps = load(name)
    out = []
    for i in range(lo, hi):
        if eps[i]["outcome"] != "success":
            continue
        env.reset(kinds=["raw"] * 6)                  # model params (egg sizes) for this episode
        for k, r in enumerate(st[i]["egg_r"]):
            g = env.egg_geom[k]
            env.m.geom_size[g] = [r, r, r * 0.026 / 0.021]
            env.m.geom_rbound[g] = r * 0.026 / 0.021
        frames = np.zeros((len(st[i]["qpos"]), 2, RES, RES, 3), np.uint8)
        for t, q in enumerate(st[i]["qpos"]):
            env.d.qpos[:] = q
            frames[t] = cams.render(env.d)
        out.append((i, frames))
    return out


def render(names=("teleop", "human")):
    """Renders every successful demo. Writes data/pixels_<name>.npy plus an index of (episode, offset, length)."""
    for name in names:
        t0 = time.time()
        eps = load(name)
        n = len(eps)
        chunks = [(name, a, min(n, a + 12)) for a in range(0, n, 12)]
        with mp.Pool(mp.cpu_count()) as pool:
            parts = [x for r in pool.imap(_render_job, chunks) for x in r]
        total = sum(len(f) for _, f in parts)
        arr = np.lib.format.open_memmap(DATA / f"pixels_{name}.npy", mode="w+", dtype=np.uint8,
                                        shape=(total, 2, RES, RES, 3))
        index, off = {}, 0
        for i, f in parts:
            arr[off:off + len(f)] = f
            index[i] = (off, len(f))
            off += len(f)
        arr.flush()
        json.dump(index, open(DATA / f"pixels_{name}.json", "w"))
        print(f"pixels {name}: {total} frames in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- dataset
def build(cfg, seed):
    """Same episode selection as policy.build (same rng draws), returning frame indices instead of features."""
    rng = np.random.default_rng(seed)
    tele, hum = load("teleop"), load("human")
    for e, j in zip(tele, range(len(tele))):
        e["_i"] = j
    for e, j in zip(hum, range(len(hum))):
        e["_i"] = j
    ok = lambda e: e["outcome"] == "success"
    raw = [e for e in tele if ok(e) and e["kind"] == "raw"]
    stone = [e for e in tele if ok(e) and e["kind"] == "stone"]
    rng.shuffle(raw); rng.shuffle(stone)
    sel = raw[: cfg["n_robot"] // 2] + stone[: cfg["n_robot"] - cfg["n_robot"] // 2]
    eps = [(e, True) for e in sel]
    if cfg["human"]:
        hs = [e for e in hum if ok(e)]
        rng.shuffle(hs)
        eps += [(e, False) for e in hs[: cfg["n_human"]]]
    idx = {n: {int(k): v for k, v in json.load(open(DATA / f"pixels_{n}.json")).items()}
           for n in ("teleop", "human") if (DATA / f"pixels_{n}.json").exists()}
    K = cfg["chunk"]
    P, F, Y, M, W = [], [], [], [], []
    for e, is_robot in eps:
        o = e["obs"]
        T = len(o)
        prop = o[:, PROPRIO + (TOUCH if cfg["touch"] else [])]
        prop = np.concatenate([prop[np.clip(np.arange(T) - h, 0, None)] for h in range(cfg["hist"])], 1)
        off, n = idx["teleop" if is_robot else "human"][e["_i"]]
        assert n == T
        src = 0 if is_robot else 1
        a = e["act"]
        y = a[np.clip(np.arange(T)[:, None] + np.arange(K)[None], 0, T - 1)].reshape(T, -1)
        mask = np.ones_like(y)
        if not is_robot and not cfg["human_grip"]:
            mask.reshape(T, K, 4)[:, :, 3] = 0
        P.append(prop.astype(np.float32))
        F.append(np.stack([np.full(T, src), off + np.arange(T)], 1))
        Y.append(y); M.append(mask)
        W.append(np.full(T, 2.0 if (is_robot and cfg["human"]) else 1.0, np.float32))
    return [np.concatenate(v) for v in (P, F, Y, M, W)]


# ---------------------------------------------------------------- model
def make_model(d_prop, d_out):
    import torch
    import torch.nn as nn

    class SpatialSoftmax(nn.Module):
        def __init__(self, h, w):
            super().__init__()
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
            self.register_buffer("xs", xs.reshape(-1)); self.register_buffer("ys", ys.reshape(-1))
            self.log_temp = nn.Parameter(torch.zeros(1))

        def forward(self, f):                          # (B, C, H, W) -> (B, 2C) expected x, y per channel
            B, C = f.shape[:2]
            p = torch.softmax(f.reshape(B, C, -1) / self.log_temp.exp(), -1)
            return torch.cat([(p * self.xs).sum(-1), (p * self.ys).sum(-1)], 1)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, KEYPOINTS, 3, 1, 1))
            self.ssm = SpatialSoftmax(RES // 4, RES // 4)

        def forward(self, x):
            return self.ssm(self.conv(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList([Encoder(), Encoder()])
            d_in = 2 * 2 * KEYPOINTS + d_prop
            self.trunk = nn.Sequential(
                nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU())
            self.act = nn.Linear(512, d_out)

        def forward(self, img, prop):                  # img: (B, 2, 3, H, W) float in [-0.5, 0.5]
            kp = [enc(img[:, c]) for c, enc in enumerate(self.enc)]
            return self.act(self.trunk(torch.cat(kp + [prop], 1)))
    return Net()


def to_input(frames):
    """uint8 (..., 2, H, W, 3) -> float (..., 2, 3, H, W) in [-0.5, 0.5]."""
    import torch
    x = torch.as_tensor(frames).float().div_(255).sub_(0.5)
    return x.movedim(-1, -3)


def _shift(x, rng):
    """Random shifts of up to SHIFT pixels with edge padding, per sample and camera."""
    import torch
    import torch.nn.functional as F
    B, C = x.shape[:2]
    flat = F.pad(x.reshape(B * C, 3, RES, RES), (SHIFT,) * 4, mode="replicate")
    d = torch.as_tensor(rng.integers(0, 2 * SHIFT + 1, (2, B * C)), device=x.device)
    ar = torch.arange(RES, device=x.device)
    rows, cols = (d[1][:, None] + ar)[:, :, None], (d[0][:, None] + ar)[:, None, :]
    out = flat[torch.arange(B * C, device=x.device)[:, None, None], :, rows, cols]    # (B*C, RES, RES, 3)
    return out.permute(0, 3, 1, 2).reshape(B, C, 3, RES, RES)


def train(cfg, seed, steps=STEPS, log=False, device="cpu"):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    P, F, Y, M, W = build(cfg, seed)
    pix = [np.load(DATA / f"pixels_{n}.npy", mmap_mode="r") if (DATA / f"pixels_{n}.npy").exists() else None
           for n in ("teleop", "human")]
    mp_, sp = P.mean(0), P.std(0) + 1e-4
    my, sy = Y.mean(0), Y.std(0) + 1e-3
    Pn, Yn, Mt = (torch.tensor(v, device=device) for v in ((P - mp_) / sp, (Y - my) / sy, M))
    model = make_model(P.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    p = W / W.sum()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        b = np.sort(rng.choice(len(P), BATCH, p=p))
        frames = np.stack([pix[s][i] for s, i in F[b]])
        img = _shift(to_input(frames).to(device), rng)
        bt = torch.as_tensor(b, device=device)
        prop = Pn[bt] + 0.02 * torch.randn(BATCH, Pn.shape[1], device=device)
        with torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):   # mixed precision on the GPU
            pa = model(img, prop)
        loss = ((pa.float() - Yn[bt]).abs() * Mt[bt]).sum() / Mt[bt].sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and step % 200 == 0:
            print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return dict(model=model.cpu(), mp=mp_, sp=sp, my=my, sy=sy, cfg=cfg)


def save(pol, path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pol, model=pol["model"].state_dict(), d_prop=len(pol["mp"]), d_out=len(pol["my"])), path)


def load_policy(path):
    import torch
    d = torch.load(path, weights_only=False)
    model = make_model(d["d_prop"], d["d_out"])
    model.load_state_dict(d["model"])
    model.eval()
    return dict(d, model=model)


class Runner:
    """Online: render both cameras, stack proprio history, predict a chunk, temporal ensembling."""

    def __init__(self, pol, cams):
        import torch
        self.torch, self.p, self.cams = torch, pol, cams
        self.reset()

    def reset(self):
        self.hist, self.pending = [], []

    def act(self, o_full, d):
        torch, cfg = self.torch, self.p["cfg"]
        prop = o_full[PROPRIO + (TOUCH if cfg["touch"] else [])]
        self.hist.insert(0, prop)
        self.hist = self.hist[: cfg["hist"]]
        while len(self.hist) < cfg["hist"]:
            self.hist.append(self.hist[-1])
        x = (np.concatenate(self.hist) - self.p["mp"]) / self.p["sp"]
        self.last_x = x
        self.frames = self.cams.render(d)
        with torch.no_grad():
            y = self.p["model"](to_input(self.frames[None]), torch.tensor(x, dtype=torch.float32)[None])[0].numpy()
        K = cfg["chunk"]
        chunk = (y * self.p["sy"] + self.p["my"]).reshape(K, 4)
        self.pending = [(c, i + 1) for c, i in self.pending if i + 1 < K] + [(chunk, 0)]
        preds = np.array([c[i] for c, i in self.pending])
        w = np.exp(-0.25 * np.arange(len(preds)))[::-1]
        return (preds * w[:, None]).sum(0) / w.sum()


def saliency(pol, frames, x):
    """|d output / d pixel| summed over the first chunk action's 4 outputs (normalised units) and colour channels:
    (2, RES, RES). The browser computes the same thing by hand (web/sim/policy.js) for its overlay."""
    import torch
    img = to_input(frames[None]).requires_grad_(True)
    y = pol["model"](img, torch.tensor(x, dtype=torch.float32)[None])[0]
    out = torch.zeros(2, RES, RES)
    for j in range(4):
        g, = torch.autograd.grad(y[j], img, retain_graph=True)
        out += g[0].abs().sum(1)
    return out.numpy()


def policy_episode(env, runner, max_steps=330, record=None):
    from .env import flat_obs
    runner.reset()
    obs = env.obs()
    opened, outcome = 0, "timeout"
    for t in range(max_steps):
        a = runner.act(flat_obs(obs), env.d)
        obs, info = env.step(a)
        if record:
            record(obs, a, info)
        st = env.egg_status()
        if st in ("cracked", "dropped"):
            outcome = st; break
        if st == "basket" and a[3] < 0:
            opened += 1
            if opened > 8:
                outcome = "success"; break
    return outcome, t + 1


def evaluate(pol, seed, n=EVAL_EPISODES, kinds=("raw", "stone")):
    import torch
    from .env import EggEnv, setup_episode
    torch.set_num_threads(1)
    env = EggEnv(seed=5000 + seed)
    rng = np.random.default_rng(9000 + seed)
    runner = Runner(pol, Cameras(env.m))
    res, steps = {k: {} for k in kinds}, []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        setup_episode(env, rng, kind)
        out, t = policy_episode(env, runner)
        res[kind][out] = res[kind].get(out, 0) + 1
        if out == "success":
            steps.append(t)
    return res, steps


# ---------------------------------------------------------------- the sweep
# Training runs on the Apple GPU in one process (mps_trainer, mixed precision, about 3.5 min a policy); a pool of CPU
# workers evaluates each policy as soon as its file appears (gain). Start both.
def _jobs():
    first = [("C", 25, 0), ("A", 25, 0)]
    return first + [j for j in ((c, n, s) for s in SEEDS for n in N_ROBOT_GRID for c in "ABCD") if j not in first]


def _path(cond, n_robot, seed):
    return POLICIES / f"{cond}_n{n_robot}_s{seed}.pt"


def _job(args):
    cond, n_robot, seed = args
    from .experiments import _summary
    path = _path(cond, n_robot, seed)
    while not path.exists():
        time.sleep(10)
    t0 = time.time()
    pol = load_policy(path)
    out = dict(cond=cond, n_robot=n_robot, seed=seed, **_summary(*evaluate(pol, seed)))
    print(f"pixels {cond} n={n_robot:3d} s={seed}  {100 * out['success']:5.1f}%  ({time.time() - t0:.0f}s eval)", flush=True)
    return out


def mps_trainer(device="mps"):
    POLICIES.mkdir(parents=True, exist_ok=True)
    for cond, n, seed in _jobs():
        path = _path(cond, n, seed)
        if path.exists():
            continue
        t0 = time.time()
        save(train(config(**CONDITIONS[cond], n_robot=n), seed, device=device), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        print(f"trained {cond} n={n} s={seed} on {device} ({time.time() - t0:.0f}s)", flush=True)


def gain(workers=6):
    from .experiments import _run
    _run(_job, _jobs(), "gain_pixels.json", lambda r: (r["cond"], r["n_robot"], r["seed"]), workers=workers)
