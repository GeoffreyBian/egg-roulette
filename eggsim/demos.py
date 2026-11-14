"""Demonstrators for egg roulette.

HumanGloveDemo: a stand-in for a person wearing our glove. Its grip strategy is taken from the
OpenGraph Labs glove data (out/human_grip.json): a gentle first squeeze that reaches its hold level in
~0.5 s, a small overshoot, grip that rises with the load it feels, and quick catch-up squeezes
when the object starts to slip (87% of the 718 real grasps had at least one). Its arm path is
minimum-jerk, the standard model of human reaching, plus retargeting error (hand -> gripper).

TeleopDemo: a stand-in for a robot teleoperator. Operators can't feel the egg, so each demo uses
one fixed squeeze the operator picked by eye. Failed demos are thrown away, as labs do.
"""
from __future__ import annotations

import json

import numpy as np

from .env import CARRY_Z, CTRL_HZ, KP_G, LEVER, Q_MIN, Q_OPEN, V_MAX, EggEnv, StepInfo, q_to_grip
from .scene import MU, ROOT

_HG = None


def human_grip_stats() -> dict:
    global _HG
    if _HG is None:
        _HG = json.loads((ROOT / "results/human_grip.json").read_text())
    return _HG


def min_jerk(p0, p1, T, t):
    s = np.clip(t / T, 0, 1)
    return p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


Q_CONTACT_NOMINAL = 0.365  # jaw angle where the pads first touch an average (21 mm) egg; +0.042 rad per mm of radius (measured)


APPROACH_RATE = 0.6     # rad/s jaw closing speed before touch
FORCE_GAIN = 0.35       # integral gain of the grip force loop (per control step)
SLIP_THRESH = 0.02      # m/s of egg motion in the gripper frame that triggers a catch-up squeeze


def squeeze_q(q_now: float, force: float) -> float:
    """Jaw command that holds `force` newtons once the pads are in contact at q_now."""
    return max(Q_MIN, q_now - force * LEVER / KP_G)


class Scripted:
    """Shared waypoint logic. Subclasses decide the grip command."""

    PINCH_DZ = 0.003    # jaw pads just above the egg's widest point (stable, and the jaws clear the carton)

    def __init__(self, env: EggEnv, rng: np.random.Generator, retarget_noise: float = 0.0):
        self.env, self.rng = env, rng
        self.noise = retarget_noise
        self.phase, self.t, self.p0 = "approach", 0, None
        self.done = False
        self.offset = rng.normal(0, retarget_noise, 3) if retarget_noise else np.zeros(3)
        self.speed = rng.uniform(0.85, 1.15)
        self.drop_xy = env.drop + rng.normal(0, 0.003, 2)

    def _goal(self) -> tuple[np.ndarray, float]:
        e = self.env
        egg = e.egg_pos()
        above = np.r_[egg[:2], CARRY_Z] + self.offset * [1, 1, 0]
        pinch = np.r_[egg[:2], egg[2] + self.PINCH_DZ] + self.offset * [1, 1, 0.3]
        basket_hi = np.r_[self.drop_xy, CARRY_Z]
        basket_lo = np.r_[self.drop_xy, 0.062]
        table = {
            "approach": (above, 1.1),
            "descend": (pinch, 0.7),
            "close": (pinch, 0.6),
            "preload": (np.r_[pinch[:2], pinch[2] + 0.012], 0.8),
            "lift": (np.r_[pinch[:2], CARRY_Z], 0.8),
            "carry": (basket_hi, 1.3),
            "lower": (basket_lo, 0.5),
            "open": (basket_lo, 0.45),
            "retreat": (np.r_[self.drop_xy, CARRY_Z], 0.5),
        }
        p, T = table[self.phase]
        return p, T * self.speed

    ORDER = ["approach", "descend", "close", "preload", "lift", "carry", "lower", "open", "retreat"]

    def act(self, info: StepInfo) -> np.ndarray:
        e = self.env
        if self.p0 is None:
            self.p0 = e.ee_target.copy()
        goal, T = self._goal()
        tt = self.t / CTRL_HZ
        ref = min_jerk(self.p0, goal, T, tt)
        dp = (ref - e.ee_target) / (V_MAX / CTRL_HZ)
        grip = self.grip(info)
        self.t += 1
        if tt >= T + 0.1:
            i = self.ORDER.index(self.phase)
            if i + 1 < len(self.ORDER):
                self.phase = self.ORDER[i + 1]
                self.t = 0
                self.p0 = e.ee_target.copy()
            else:
                self.done = True
        return np.r_[np.clip(dp, -1, 1), grip]

    def grip(self, info: StepInfo) -> float:
        raise NotImplementedError


class HumanGloveDemo(Scripted):
    """Grip strategy from the human glove data: start gentle, track the load, catch up on slip."""

    F_START = 0.9          # first squeeze, N (A): enough for a light egg
    SAFETY = 1.35          # grip / minimum-to-hold ratio people keep (A, typical 1.2-1.6)

    def __init__(self, env, rng, retarget_noise=0.0015, heavy_handed=False):
        super().__init__(env, rng, retarget_noise)
        if heavy_handed:                   # starts firm and keeps a big margin
            self.F_START, self.SAFETY = 1.7, 1.8
        hg = human_grip_stats()
        lo, hi = hg["rise_s_iqr"]
        self.rise = float(np.clip(rng.uniform(lo, hi) * 0.6, 0.15, 0.6))   # robot-speed version of the human rise
        olo, ohi = hg["overshoot_iqr"]
        self.overshoot = float(np.clip(rng.uniform(olo, ohi), 1.0, 1.3))
        self.f_target = self.F_START
        self.t_close = 0

    def grip(self, info: StepInfo) -> float:
        ph = self.phase
        q = float(self.env.d.qpos[self.env.arm_qpos[5]])
        if ph in ("approach", "descend", "open", "retreat"):
            self.qc, self.in_contact, self.lost = Q_OPEN, False, 0
            return -1.0
        if not getattr(self, "in_contact", False):
            # close gently until the pads report a squeeze
            self.qc = max(Q_MIN, min(self.qc, q + 0.02) - APPROACH_RATE / CTRL_HZ)
            if info.grip_n > 0.1:
                self.in_contact = True
                self.qc = squeeze_q(q, 0.3)
            return q_to_grip(self.qc)
        if ph == "close":
            self.t_close += 1
            s = min(1.0, self.t_close / (self.rise * CTRL_HZ))
            target = self.F_START * max(0.3, s * (1 + (self.overshoot - 1) * np.sin(np.pi * s)))
        else:
            # lift / carry / lower: grip follows the load it feels, plus catch-up squeezes on slip
            need = info.shear_n / (2 * MU)
            self.f_target = max(self.f_target, self.SAFETY * need, self.F_START)
            near_limit = info.shear_n > 0.5 * 2 * MU * max(info.grip_n, 1e-3)   # load near what this grip can hold
            if info.slip > SLIP_THRESH and near_limit:
                self.f_target *= 1.12
            self.f_target = min(self.f_target, 5.0)
            target = self.f_target
        self.lost = self.lost + 1 if info.grip_n < 0.05 else 0
        if self.lost > 3:                  # lost the egg: close in again
            self.qc -= APPROACH_RATE / CTRL_HZ
        else:                              # integral force control on the jaw command
            self.qc -= FORCE_GAIN * (target - info.grip_n) * LEVER / KP_G
        self.qc = float(np.clip(self.qc, max(Q_MIN, q - 0.4), q + 0.05))
        return q_to_grip(self.qc)


class TeleopDemo(Scripted):
    """Operator picks one squeeze by eye (a jaw angle past where an average egg is touched) and keeps
    it. Can't feel the egg, and can't see sub-millimetre differences in egg size."""

    def __init__(self, env, rng, retarget_noise=0.0):
        super().__init__(env, rng, retarget_noise=0.0)
        self.depth = float(rng.uniform(0.03, 0.30))
        self.t_close = 0

    def grip(self, info: StepInfo) -> float:
        if self.phase in ("approach", "descend", "open", "retreat"):
            return -1.0
        self.t_close += 1
        s = min(1.0, self.t_close / (0.3 * CTRL_HZ))
        q_cmd = Q_OPEN + (Q_CONTACT_NOMINAL - self.depth - Q_OPEN) * s
        return q_to_grip(q_cmd)


def run_episode(env: EggEnv, demo: Scripted, max_steps: int = 260, record=None):
    """Runs one egg. Returns (steps, outcome). record(obs, action, info) is called each step."""
    obs = env.obs()
    info = StepInfo()
    outcome = None
    for t in range(max_steps):
        a = demo.act(info)
        if record:
            record(obs, a, info)
        obs, info = env.step(a)
        st = env.egg_status()
        if st == "cracked":
            outcome = "cracked"
            break
        if st == "dropped" and demo.phase in ("preload", "lift", "carry", "lower"):
            outcome = "dropped"
            break
        if demo.done:
            break
    if outcome is None:
        for _ in range(20):
            obs, info = env.step(np.array([0, 0, 0.3, -1.0]))
        st = env.egg_status()
        outcome = {"basket": "success", "cracked": "cracked"}.get(st, "dropped" if st != "carton" else "missed")
    return t + 1, outcome


def diverse_kind(rng: np.random.Generator) -> str:
    """A random egg from a spread of weights and fragilities. 130-170 g is held out so the 150 g
    'medium' test egg stays unseen."""
    while True:
        m = rng.uniform(0.05, 0.36)
        if not 0.13 <= m <= 0.17:
            break
    floor = 11 * m + 0.5                   # crack force must leave room to hold it
    c = rng.uniform(max(1.6, floor), 4.5) if (floor < 4.5 and rng.random() < 0.6) else float("inf")
    return f"m{m:.3f}_c{c:.2f}"
