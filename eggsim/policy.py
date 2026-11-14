"""Behaviour cloning: an ACT-style action-chunking MLP, trained per configuration, tested on fresh eggs.

A config is a dict of switches over BASE. The four headline conditions:
  A  robot demos only            teleop demos, no touch input
  B  + human video               teleop + human demos; the human's grip force is unknown from video, so
                                 the grip output trains on teleop only; no touch input
  C  + human touch               teleop + human glove demos with grip force; touch input
  D  robot touch sensor only     teleop demos with a pad sensor on the robot; no human data
"""
from __future__ import annotations

import numpy as np

from .collect import PHASES, load

TOUCH_DIMS = 3
EVAL_EPISODES = 60                   # half raw, half stone
MAX_STEPS = 330
GEN_SUITES = ["raw_fragile", "medium", "stone_light", "stone_heavy"]

BASE = dict(touch=True, human=True, human_grip=True, n_robot=25, n_human=150, hist=2, chunk=8,
            aux_phase=False, aux_kind=False, aux_slip=False, success_cond=False, unfiltered=False, steps=12000,
            touch_lag=0, miscal=0.0, quality_top=False, touch_mask=(1, 1, 1), proxy=False, test_proxy=False,
            touch_rate=1, heavy_frac=0.0, heavy_label=False, gen=False, human_src="human")

CONDITIONS = {
    "A": dict(label="Robot demos only", touch=False, human=False),
    "B": dict(label="+ human video, no touch", touch=False, human_grip=False),
    "C": dict(label="+ human demos with touch", touch=True),
    "D": dict(label="Touch sensor on robot, robot demos only", touch=True, human=False),
}


def config(**kw) -> dict:
    cfg = dict(BASE, **{k: v for k, v in kw.items() if k != "label"})
    unknown = set(cfg) - set(BASE)
    assert not unknown, unknown
    return cfg


# ---------------------------------------------------------------- sensing transforms
def servo_estimate(o, rng):
    """Squeeze a servo could report from its own position error, written into touch dim 0; no load, no slip.
    o: (T, 20) with columns 3 = jaw angle, 16 = previous grip command, 17-19 = touch."""
    from .env import KP_G, LEVER, grip_to_q
    q, g = o[:, 3], o[:, 16]
    q_cmd = np.array([grip_to_q(x) for x in g])
    f = np.clip(KP_G * (q - q_cmd), 0, None) / LEVER * (g >= 0)
    f = np.clip(f + rng.normal(0, 0.3, len(f)), 0, None)        # current sensing noise ~0.3 N
    out = o.copy()
    out[:, -3] = f / 3.0
    out[:, -2:] = 0
    return out


def transform(o, cfg, rng, test=False):
    o = o.copy()
    if cfg["proxy"] or (test and cfg["test_proxy"]):
        o = servo_estimate(o, rng)
    o[:, -TOUCH_DIMS:] *= np.asarray(cfg["touch_mask"], np.float32)
    return o


def hold(o, rate):
    """Sample-and-hold the touch channels at a lower rate."""
    if rate <= 1:
        return o
    o = o.copy()
    o[:, -TOUCH_DIMS:] = o[(np.arange(len(o)) // rate) * rate, -TOUCH_DIMS:]
    return o


def featurize(o, touch, hist):
    o = o if touch else o[:, :-TOUCH_DIMS]
    T = len(o)
    return np.concatenate([o[np.clip(np.arange(T) - h, 0, None)] for h in range(hist)], 1).astype(np.float32)


# ---------------------------------------------------------------- dataset
def build(cfg, seed):
    rng = np.random.default_rng(seed)
    tele, hum = load("teleop"), load(cfg["human_src"])
    ok = lambda e: e["outcome"] == "success"
    raw = [e for e in tele if ok(e) and e["kind"] == "raw"]
    stone = [e for e in tele if ok(e) and e["kind"] == "stone"]
    rng.shuffle(raw); rng.shuffle(stone)
    sel = raw[: cfg["n_robot"] // 2] + stone[: cfg["n_robot"] - cfg["n_robot"] // 2]
    eps = [(e, True) for e in sel]
    if cfg["success_cond"] or cfg["unfiltered"]:
        # the failed attempts the same operators made on the way to those successes
        fails = [e for e in tele if not ok(e)]
        rng.shuffle(fails)
        eps += [(e, True) for e in fails[: 2 * cfg["n_robot"]]]
    if cfg["human"]:
        hs = [e for e in hum if ok(e)]
        rng.shuffle(hs)
        if cfg["quality_top"]:
            # "quality" filter: fewest slip events, then shortest episode
            hs = sorted(hs, key=lambda e: (int(e["slip"].sum()), len(e["act"])))
        n_heavy = int(round(cfg["heavy_frac"] * cfg["n_human"]))
        if n_heavy:
            hv = [dict(e, heavy=True) for e in load("human_heavy") if ok(e)]
            rng.shuffle(hv)
            eps += [(e, False) for e in hv[:n_heavy]]
        eps += [(e, False) for e in hs[: cfg["n_human"] - n_heavy]]
        if cfg["success_cond"] or cfg["unfiltered"]:
            hf = [e for e in hum if not ok(e)]
            rng.shuffle(hf)
            eps += [(e, False) for e in hf[: cfg["n_human"] // 2]]
    K = cfg["chunk"]
    X, Y, M, W, PH, KD, SL = [], [], [], [], [], [], []
    for e, is_robot in eps:
        o = e["obs"].copy()
        if not is_robot and cfg["touch_lag"]:
            # glove and camera clocks out of sync: touch arrives touch_lag steps late
            o[:, -TOUCH_DIMS:] = o[np.clip(np.arange(len(o)) - cfg["touch_lag"], 0, None), -TOUCH_DIMS:]
        if not is_robot and cfg["miscal"]:
            # glove force calibration off by a per-session factor
            o[:, -TOUCH_DIMS:-1] *= rng.uniform(1 - cfg["miscal"], 1 + cfg["miscal"])
        o = hold(transform(o, cfg, rng), cfg["touch_rate"])
        x = featurize(o, cfg["touch"], cfg["hist"])
        if cfg["heavy_label"]:
            x = np.concatenate([x, np.full((len(x), 1), 1.0 if e.get("heavy") else 0.0, np.float32)], 1)
        if cfg["success_cond"]:
            x = np.concatenate([x, np.full((len(x), 1), 1.0 if ok(e) else 0.0, np.float32)], 1)
        a = e["act"]
        T = len(a)
        idx = np.clip(np.arange(T)[:, None] + np.arange(K)[None], 0, T - 1)
        y = a[idx].reshape(T, -1)
        mask = np.ones_like(y)
        if not is_robot and not cfg["human_grip"]:
            mask.reshape(T, K, 4)[:, :, 3] = 0       # grip force unknown from video
        X.append(x); Y.append(y); M.append(mask)
        # weight robot demos up so that a handful of them still matter next to 150 human demos
        W.append(np.full(T, 2.0 if (is_robot and cfg["human"]) else 1.0, np.float32))
        PH.append(e["phase"].astype(np.int64))
        KD.append(np.full(T, 1.0 if e["kind"] == "stone" else 0.0, np.float32))
        SL.append(e["slip"].astype(np.float32))
    return [np.concatenate(v) for v in (X, Y, M, W, PH, KD, SL)]


# ---------------------------------------------------------------- model
def make_model(d_in, d_out, n_aux=0):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU())
            self.act = nn.Linear(512, d_out)
            self.aux = nn.Linear(512, n_aux) if n_aux else None

        def forward(self, x):
            h = self.trunk(x)
            return self.act(h), (self.aux(h) if self.aux is not None else None)
    return Net()


def train(cfg, seed):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    X, Y, M, W, PH, KD, SL = build(cfg, seed)
    mx, sx = X.mean(0), X.std(0) + 1e-4
    n_flags = int(cfg["success_cond"]) + int(cfg["heavy_label"])
    if n_flags:
        mx[-n_flags:], sx[-n_flags:] = 0.0, 1.0
    my, sy = Y.mean(0), Y.std(0) + 1e-3
    Xn, Yn, Mt = torch.tensor((X - mx) / sx), torch.tensor((Y - my) / sy), torch.tensor(M)
    PHt, KDt, SLt = torch.tensor(PH), torch.tensor(KD), torch.tensor(SL)
    n_aux = len(PHASES) * cfg["aux_phase"] + cfg["aux_kind"] + cfg["aux_slip"]
    model = make_model(X.shape[1], Y.shape[1], n_aux)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=cfg["steps"])
    p = W / W.sum()
    rng = np.random.default_rng(seed)
    for _ in range(cfg["steps"]):
        b = torch.tensor(rng.choice(len(Xn), 512, p=p))
        xb = Xn[b] + 0.02 * torch.randn(512, Xn.shape[1])
        if n_flags:
            xb[:, -n_flags:] = Xn[b, -n_flags:]
        pa, px = model(xb)
        loss = ((pa - Yn[b]).abs() * Mt[b]).sum() / Mt[b].sum()
        j = 0
        if cfg["aux_phase"]:
            loss = loss + 0.3 * F.cross_entropy(px[:, j:j + len(PHASES)], PHt[b]); j += len(PHASES)
        if cfg["aux_kind"]:
            loss = loss + 0.3 * F.binary_cross_entropy_with_logits(px[:, j], KDt[b]); j += 1
        if cfg["aux_slip"]:
            loss = loss + 0.3 * F.binary_cross_entropy_with_logits(px[:, j], SLt[b], pos_weight=torch.tensor(5.0))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return dict(model=model, mx=mx, sx=sx, my=my, sy=sy, cfg=cfg, n_aux=n_aux)


def save(pol, path):
    import torch
    torch.save(dict(pol, model=pol["model"].state_dict(), d_in=len(pol["mx"]), d_out=len(pol["my"])), path)


def load_policy(path):
    import torch
    d = torch.load(path, weights_only=False)
    model = make_model(d["d_in"], d["d_out"], d["n_aux"])
    model.load_state_dict(d["model"])
    model.eval()
    return dict(d, model=model)


class Runner:
    """Runs a trained policy online: history stacking, action chunks and ACT-style temporal ensembling."""

    def __init__(self, pol):
        import torch
        self.torch, self.p = torch, pol
        self.reset()

    def reset(self):
        self.hist, self.pending = [], []
        self.t, self.held = 0, None
        self.rng = np.random.default_rng(12345)

    def act(self, o_full):
        torch, cfg = self.torch, self.p["cfg"]
        o_full = transform(o_full[None], cfg, self.rng, test=True)[0]
        if cfg["touch_rate"] > 1:
            if self.t % cfg["touch_rate"] == 0 or self.held is None:
                self.held = o_full[-TOUCH_DIMS:].copy()
            o_full = np.r_[o_full[:-TOUCH_DIMS], self.held]
        self.t += 1
        o = o_full if cfg["touch"] else o_full[:-TOUCH_DIMS]
        self.hist.insert(0, o)
        self.hist = self.hist[: cfg["hist"]]
        while len(self.hist) < cfg["hist"]:
            self.hist.append(self.hist[-1])
        x = np.concatenate(self.hist).astype(np.float32)
        if cfg["heavy_label"]:
            x = np.r_[x, 0.0]                      # ask for a careful demonstrator
        if cfg["success_cond"]:
            x = np.r_[x, 1.0]                      # ask for a successful episode
        with torch.no_grad():
            y = self.p["model"](torch.tensor((x - self.p["mx"]) / self.p["sx"])[None])[0][0].numpy()
        K = cfg["chunk"]
        chunk = (y * self.p["sy"] + self.p["my"]).reshape(K, 4)
        # temporal ensembling: average this step's predictions from every chunk still in flight
        self.pending = [(c, i + 1) for c, i in self.pending if i + 1 < K] + [(chunk, 0)]
        preds = np.array([c[i] for c, i in self.pending])
        w = np.exp(-0.25 * np.arange(len(preds)))[::-1]
        return (preds * w[:, None]).sum(0) / w.sum()


def policy_episode(env, runner, max_steps=MAX_STEPS, record=None):
    """One egg with a learned policy. Outcome: success | cracked | dropped | timeout."""
    from .env import flat_obs
    runner.reset()
    obs = env.obs()
    opened_in_basket = 0
    outcome = "timeout"
    for t in range(max_steps):
        a = runner.act(flat_obs(obs))
        obs, info = env.step(a)
        if record:
            record(obs, a, info)
        st = env.egg_status()
        if st == "cracked":
            outcome = "cracked"; break
        if st == "dropped":
            outcome = "dropped"; break
        if st == "basket" and a[3] < 0:
            opened_in_basket += 1
            if opened_in_basket > 8:
                outcome = "success"; break
    return outcome, t + 1


def evaluate(pol, seed, n=EVAL_EPISODES, kinds=("raw", "stone"), env_kw=None):
    """Success counts on n fresh eggs, alternating kinds. Returns {kind: {outcome: count}}, steps per success."""
    from .env import EggEnv, setup_episode
    env = EggEnv(seed=5000 + seed, **(env_kw or {}))
    rng = np.random.default_rng(9000 + seed)
    runner = Runner(pol)
    res = {k: {} for k in kinds}
    steps = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        setup_episode(env, rng, kind)
        out, t = policy_episode(env, runner)
        res[kind][out] = res[kind].get(out, 0) + 1
        if out == "success":
            steps.append(t)
    return res, steps
