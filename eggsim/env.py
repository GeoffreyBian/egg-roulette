"""Egg roulette: an SO-101 moves look-alike eggs from a carton to a basket.

Two kinds of egg look identical to the camera:
  raw   : 60 g, cracks if squeezed harder than 2.2 N
  stone : 300 g painted stone, slips out of the SO-101's jaws unless squeezed harder than ~3.5 N (measured,
          see docs/physics.md)

No single squeeze works for both. A robot that can feel (pad normal force, load, slip) can start gentle
and tighten when the egg turns out to be heavy. One that can't, has to guess.

Control runs at 30 Hz, physics at 500 Hz. Action = [dx, dy, dz, grip] in [-1, 1]: dx..dz move the pinch
point target (V_MAX m/s, solved to joint targets by damped least-squares IK), grip commands the jaw angle
from fully open (-1) to fully closed (+1), as a LeRobot SO-101 action does.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import (BASKET_C, BASKET_HALF, CARTON_FLOOR, CTRL_HZ, EGG_H, EGG_R, EGG_R_SD, MU, PAD_TORSION, PARK,
                    SIM_DT, SLOTS, build_model)

SUBSTEPS = int(round(1 / CTRL_HZ / SIM_DT))

MASS = {"raw": 0.060, "stone": 0.300,
        # unseen eggs, only used to test generalization
        "raw_fragile": 0.060, "medium": 0.150, "stone_light": 0.180, "stone_heavy": 0.400}
CRACK_N = 2.2                        # raw egg cracks above this grip force (A, scaled to the SO-101)
CRACK = {"raw": CRACK_N, "raw_fragile": 1.6, "medium": 3.5}   # kinds not listed never crack


def egg_props(kind: str) -> tuple[float, float]:
    """(mass kg, crack N) for a named kind, or a custom one written 'm0.120_c3.10' (c inf = never cracks)."""
    if kind.startswith("m") and "_c" in kind:
        m, c = kind[1:].split("_c")
        return float(m), float(c)
    return MASS[kind], CRACK.get(kind, float("inf"))


LEVER = 0.060                        # effective jaw joint to pad lever, m (calibrated: measured grip vs torque)
V_MAX = 0.22                         # pinch point speed limit, m/s
DROP_SPOTS = [BASKET_C + np.array([dx, dy]) for dy in (0.045, 0.0, -0.045) for dx in (-0.025, 0.025)]
PICK_ORDER = [0, 1, 2, 3, 4, 5]      # near row first: the carry path never passes over an egg
CARRY_Z = 0.085                      # pinch point height when carrying
HOME = np.array([0.18, -0.02, 0.095])
WS_LO = np.array([0.12, -0.24, 0.025])
WS_HI = np.array([0.34, 0.24, 0.105])
READY = [0.0, -0.4, 0.3, 1.5]        # pan, lift, elbow, wrist_flex seed for IK (the straight pose is singular)
Q_OPEN = 0.75                        # open jaw angle: 74 mm between fingertips
Q_MIN = -0.10                        # grip = +1 commands this jaw angle (fully closed)
PINCH_LOCAL = np.array([0.0117, -0.0002, -0.0830])  # gripper frame: egg centre when pinched
KP_G = 1.2                           # jaw P gain, N m / rad: 1 N at the pads per ~0.05 rad past contact
KD_G = 0.03
TAU_LIM = 0.35                       # jaw torque limit, N m (~5.8 N at the pads)
W_ORI = 0.006                        # IK weight on keeping the gripper pointing down
ROLL = 1.57 - np.pi                  # wrist roll: moving jaw on the -y side


@dataclass
class EggState:
    kind: str
    slot: int
    cracked: bool = False
    peak_grip: float = 0.0


@dataclass
class StepInfo:
    grip_n: float = 0.0               # squeeze: smaller of the two jaws' summed normal forces, N
    shear_n: float = 0.0              # load: upward force the jaws exert on the egg, N
    slip: float = 0.0                 # egg speed relative to the gripper, m/s
    event: str | None = None


class EggEnv:
    """One target egg at a time. reset() puts eggs in the carton; set_target(k) picks which one.

    mu and mass_scale are domain-randomization knobs: pad friction, and a multiplier on every egg's mass.
    """

    def __init__(self, n_eggs: int = 6, seed: int = 0, vision_noise: float = 0.003, render: bool = False,
                 mu: float = MU, mass_scale: float = 1.0):
        self.m = build_model(n_eggs, render=render)
        self.d = mujoco.MjData(self.m)
        self.n_eggs = n_eggs
        self.rng = np.random.default_rng(seed)
        self.vision_noise = vision_noise
        self.mass_scale = mass_scale
        m = self.m
        self.site = m.site("gripperframe").id
        arm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
        self.arm_dofs = [m.joint(n).dofadr[0] for n in arm]
        self.arm_qpos = [m.joint(n).qposadr[0] for n in arm + ["wrist_roll", "gripper"]]
        self.arm_range = [m.jnt_range[m.joint(n).id] for n in arm]
        self.grip_act = m.actuator("gripper").id
        self.grip_dof = m.joint("gripper").dofadr[0]
        self.gripper_body = m.body("gripper").id
        self.moving_jaw = m.body("moving_jaw_so101_v1").id
        self.jaw_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in (self.gripper_body, self.moving_jaw)}
        self.egg_body = [m.body(f"egg{k}").id for k in range(n_eggs)]
        self.egg_geom = [m.geom(f"egg{k}").id for k in range(n_eggs)]
        self.egg_qadr = [m.joint(f"egg{k}").qposadr[0] for k in range(n_eggs)]
        self.egg_dadr = [m.joint(f"egg{k}").dofadr[0] for k in range(n_eggs)]
        self.debris = {k: [m.body(f"{p}{k}").id for p in ("yolk", "white", "shellA", "shellB")]
                       for k in range(n_eggs)}
        self.sensor_adr = {n: m.sensor(n).adr[0] for n in ("touch_fixed", "touch_moving", "gripper_torque")}
        self.set_friction(mu)
        self._ikd = mujoco.MjData(m)
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self.eggs: list[EggState] = []
        self.target = 0
        self.drop = BASKET_C.copy()
        self.ee_target = HOME.copy()
        self.prev_action = np.zeros(4)
        self.t = 0
        self.last = StepInfo()

    def set_friction(self, mu: float):
        self.mu = mu
        for g in self.jaw_geoms:
            self.m.geom_friction[g] = [mu, PAD_TORSION, 0.002]

    # ---------- setup ----------
    def reset(self, kinds: list[str] | None = None, jitter: float = 0.002, present: list[int] | None = None):
        """present: which eggs sit in the carton (others stay parked off stage). Default: all."""
        m, d = self.m, self.d
        for k in range(self.n_eggs):
            m.body_gravcomp[self.egg_body[k]] = 0
            m.geom_contype[self.egg_geom[k]], m.geom_conaffinity[self.egg_geom[k]] = 1, 1
            for b in self.debris[k]:
                m.body_gravcomp[b] = 1
                for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
                    m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
        if kinds is None:
            kinds = list(self.rng.choice(["raw", "stone"], self.n_eggs))
        self.eggs, self.egg_r = [], []
        for k in range(self.n_eggs):
            r = EGG_R + float(np.clip(self.rng.normal(0, EGG_R_SD), -2 * EGG_R_SD, 2 * EGG_R_SD))
            self.egg_r.append(r)
            g = self.egg_geom[k]
            m.geom_size[g] = [r, r, r * EGG_H / EGG_R]
            m.geom_rbound[g] = r * EGG_H / EGG_R
            mass = egg_props(kinds[k])[0] * self.mass_scale
            m.body_mass[self.egg_body[k]] = mass
            # solid ellipsoid inertia
            m.body_inertia[self.egg_body[k]] = mass / 5 * np.array(
                [EGG_R ** 2 + EGG_H ** 2, EGG_R ** 2 + EGG_H ** 2, 2 * EGG_R ** 2])
        mujoco.mj_setConst(m, d)          # recompute derived constants after the mass changes
        mujoco.mj_resetData(m, d)
        for k in range(self.n_eggs):
            s = k % len(SLOTS)
            x, y = SLOTS[s] + self.rng.uniform(-jitter, jitter, 2)
            q = self.egg_qadr[k]
            d.qpos[q:q + 3] = [x, y, CARTON_FLOOR + self.egg_r[k] * EGG_H / EGG_R + 0.0005]
            if present is not None and k not in present:
                d.qpos[q:q + 3] = PARK + [k * 0.2, 0, 0]
            d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
            self.eggs.append(EggState(kind=kinds[k], slot=s))
        for a, v in zip(self.arm_qpos[:4], READY):
            d.qpos[a] = v
        d.qpos[self.arm_qpos[4]] = ROLL
        d.qpos[self.arm_qpos[5]] = Q_OPEN
        self.ee_target = HOME.copy()
        q = self._ik(HOME, iters=200)
        d.qpos[self.arm_qpos[:4]] = q
        d.ctrl[:4] = q
        d.ctrl[4] = ROLL
        d.ctrl[self.grip_act] = 0.2
        mujoco.mj_forward(m, d)
        for _ in range(150):              # let the eggs settle on their supports
            mujoco.mj_step(m, d)
        self.prev_action = np.zeros(4)
        self.t = 0
        self.target = 0
        self.last = StepInfo()
        return self.obs()

    def reset_arm(self):
        """Put the arm back in the start pose without touching the eggs (between eggs in a full carton)."""
        m, d = self.m, self.d
        q = self._ik(HOME, iters=200)
        d.qpos[self.arm_qpos[:4]] = q
        d.qpos[self.arm_qpos[4]] = ROLL
        d.qpos[self.arm_qpos[5]] = Q_OPEN
        d.qvel[[m.jnt_dofadr[m.joint(n).id] for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                      "wrist_roll", "gripper")]] = 0
        d.ctrl[:4] = q
        d.ctrl[4] = ROLL
        mujoco.mj_forward(m, d)
        for _ in range(60):
            d.ctrl[self.grip_act] = np.clip(KP_G * (Q_OPEN - d.qpos[self.arm_qpos[5]]), -TAU_LIM, TAU_LIM)
            mujoco.mj_step(m, d)
        self.ee_target = HOME.copy()
        self.prev_action = np.zeros(4)
        self.last = StepInfo()

    def set_target(self, k: int, drop: np.ndarray | None = None):
        self.target = k
        self.drop = BASKET_C.copy() if drop is None else np.asarray(drop, float)
        self.t = 0

    # ---------- kinematics ----------
    def ee(self, dd=None) -> np.ndarray:
        dd = self.d if dd is None else dd
        b = self.gripper_body
        return dd.xpos[b] + dd.xmat[b].reshape(3, 3) @ PINCH_LOCAL

    def _ik(self, p_des: np.ndarray, iters: int = 40) -> np.ndarray:
        """Damped least squares on [pinch position, approach axis = down] over the 4 arm joints."""
        m, dd = self.m, self._ikd
        dd.qpos[:] = self.d.qpos
        jacp, jacr = self._jacp, self._jacr
        down = np.array([0, 0, -1.0])
        for _ in range(iters):
            mujoco.mj_kinematics(m, dd)
            mujoco.mj_comPos(m, dd)
            p = self.ee(dd)
            a = dd.site_xmat[self.site].reshape(3, 3)[:, 0]
            e = np.r_[p_des - p, W_ORI * np.cross(a, down)]
            mujoco.mj_jac(m, dd, jacp, jacr, p, self.gripper_body)
            J = np.r_[jacp[:, self.arm_dofs], W_ORI * jacr[:, self.arm_dofs]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            dq = np.clip(dq, -0.15, 0.15)
            for i, adr in enumerate(self.arm_qpos[:4]):
                lo, hi = self.arm_range[i]
                dd.qpos[adr] = np.clip(dd.qpos[adr] + dq[i], lo, hi)
            if np.linalg.norm(e[:3]) < 2e-4:
                break
        return np.array([dd.qpos[a] for a in self.arm_qpos[:4]])

    # ---------- sensing ----------
    def egg_pos(self, k: int | None = None) -> np.ndarray:
        k = self.target if k is None else k
        return self.d.xpos[self.egg_body[k]].copy()

    def touch(self, k: int | None = None) -> StepInfo:
        """Ground-truth contact forces between the jaws and egg k: squeeze, load and slip speed.

        Uses mj_contactForce on every active contact, rotated out of the contact frame. The native `touch`
        sensors on the pad sites (see native_touch) report the same normal forces; see docs/physics.md.
        """
        k = self.target if k is None else k
        m, d = self.m, self.d
        g_egg = self.egg_geom[k]
        f6 = np.zeros(6)
        normal = [0.0, 0.0]
        force = np.zeros(3)
        n_c = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == g_egg:
                other, sign = c.geom2, -1.0       # contact force acts on geom2 by convention
            elif c.geom2 == g_egg:
                other, sign = c.geom1, 1.0
            else:
                continue
            if other not in self.jaw_geoms:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            frame = c.frame.reshape(3, 3)     # rows: normal, tangent 1, tangent 2
            normal[int(m.geom_bodyid[other] == self.moving_jaw)] += abs(f6[0])
            force += sign * (f6[:3] @ frame)
            n_c += 1
        slip = 0.0
        if n_c:
            b = self.gripper_body
            R = d.xmat[b].reshape(3, 3)
            v_egg, v_g = np.zeros(6), np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.egg_body[k], v_egg, 0)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v_g, 0)
            r = d.xpos[self.egg_body[k]] - d.xpos[b]
            v_point = v_g[3:] + np.cross(v_g[:3], r)          # gripper velocity at the egg's centre
            slip = float(np.linalg.norm(R.T @ (v_egg[3:] - v_point)))
        # a one-sided push isn't a pinch: the squeeze is what both jaws push with
        return StepInfo(grip_n=min(normal), shear_n=float(max(0.0, force[2])), slip=slip)

    def native_touch(self) -> tuple[float, float]:
        """MuJoCo touch sensors on the fixed and moving pad sites, N."""
        a = self.sensor_adr
        return float(self.d.sensordata[a["touch_fixed"]]), float(self.d.sensordata[a["touch_moving"]])

    def servo_torque(self) -> float:
        return float(self.d.sensordata[self.sensor_adr["gripper_torque"]])

    def obs(self) -> dict:
        ti = self.last
        return dict(
            ee=self.ee(),
            grip_q=np.array([self.d.qpos[self.arm_qpos[5]]]),
            egg=self.egg_pos() + self.rng.normal(0, self.vision_noise, 3),
            basket=np.r_[self.drop, 0.0],
            prev=self.prev_action.copy(),
            touch=np.array([ti.grip_n / 3.0, ti.shear_n / 3.0, min(ti.slip, 0.3) / 0.1]),
        )

    # ---------- stepping ----------
    def step(self, action: np.ndarray, on_substep=None) -> tuple[dict, StepInfo]:
        m, d = self.m, self.d
        a = np.clip(np.asarray(action, float), -1, 1)
        self.ee_target = np.clip(self.ee_target + a[:3] * V_MAX / CTRL_HZ, WS_LO, WS_HI)
        # keep the target close to the real pinch point so the arm can't wind up
        self.ee_target = self.ee() + np.clip(self.ee_target - self.ee(), -0.03, 0.03)
        d.ctrl[:4] = self._ik(self.ee_target)
        d.ctrl[4] = ROLL
        qg = self.arm_qpos[5]
        q_cmd = grip_to_q(a[3])
        info = StepInfo()
        for _ in range(SUBSTEPS):
            # jaw servo: P loop with a softened gain and a torque limit, closed at the physics rate
            d.ctrl[self.grip_act] = np.clip(KP_G * (q_cmd - d.qpos[qg]) - KD_G * d.qvel[self.grip_dof],
                                            -TAU_LIM, TAU_LIM)
            mujoco.mj_step(m, d)
            ti = self.touch()
            # a real pad sensor averages over its sample period: report the mean over the substeps
            info.grip_n += ti.grip_n / SUBSTEPS
            info.shear_n += ti.shear_n / SUBSTEPS
            info.slip += ti.slip / SUBSTEPS
            if on_substep:
                on_substep(ti)
        info.event = self._check_egg(info)   # shells crack under sustained load: judged on the 33 ms average
        self.prev_action = a
        self.t += 1
        self.last = info
        return self.obs(), info

    def _check_egg(self, info: StepInfo) -> str | None:
        e = self.eggs[self.target]
        if e.cracked:
            return None
        e.peak_grip = max(e.peak_grip, info.grip_n)
        if info.grip_n > egg_props(e.kind)[1]:
            self._crack(self.target)
            return "crack"
        return None

    def _crack(self, k: int):
        """Swap the egg for yolk, white and shell pieces at the same place and velocity."""
        m, d = self.m, self.d
        self.eggs[k].cracked = True
        p = self.egg_pos(k)
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.egg_body[k], v, 0)
        offs = [(0, 0, -0.012), (0.004, -0.003, -0.02), (0.016, 0.01, -0.005), (-0.015, -0.008, -0.004)]
        for i, (b, o) in enumerate(zip(self.debris[k], offs)):
            jq = m.body_jntadr[b]
            qa, da = m.jnt_qposadr[jq], m.jnt_dofadr[jq]
            d.qpos[qa:qa + 3] = p + np.array(o)
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            d.qvel[da:da + 3] = v[3:] + (self.rng.normal(0, 0.15, 3) if i else 0)
            m.body_gravcomp[b] = 0
            m.body_contype[b], m.body_conaffinity[b] = 1, 1      # broad-phase flags are cached per body
            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 1
        m.body_gravcomp[self.egg_body[k]] = 1
        m.geom_contype[self.egg_geom[k]], m.geom_conaffinity[self.egg_geom[k]] = 0, 0
        qa = self.egg_qadr[k]
        d.qpos[qa:qa + 3] = [PARK[0] + k * 0.2, PARK[1] - 1, PARK[2]]
        d.qvel[self.egg_dadr[k]:self.egg_dadr[k] + 6] = 0
        mujoco.mj_forward(m, d)

    def egg_status(self, k: int | None = None) -> str:
        """'cracked' | 'basket' | 'dropped' | 'carton' | 'held'"""
        k = self.target if k is None else k
        e = self.eggs[k]
        if e.cracked:
            return "cracked"
        p = self.egg_pos(k)
        if np.all(np.abs(p[:2] - BASKET_C) < BASKET_HALF) and p[2] < 0.07:
            return "basket"
        if np.linalg.norm(p[:2] - SLOTS[e.slot]) < 0.02 and p[2] < 0.05:
            return "carton"
        if p[2] < EGG_H + 0.012:
            return "dropped"
        return "held"


def grip_to_q(g: float) -> float:
    return Q_OPEN + (Q_MIN - Q_OPEN) * (np.clip(g, -1, 1) + 1) / 2


def q_to_grip(q: float) -> float:
    return float(np.clip(2 * (q - Q_OPEN) / (Q_MIN - Q_OPEN) - 1, -1, 1))


def flat_obs(o: dict, touch: bool = True) -> np.ndarray:
    parts = [o["ee"], o["grip_q"], o["egg"], o["egg"] - o["ee"], o["basket"] - o["ee"], o["prev"]]
    if touch:
        parts.append(o["touch"])
    return np.concatenate(parts).astype(np.float32)


def setup_episode(env: EggEnv, rng: np.random.Generator, kind: str) -> int:
    """A mid-carton moment: target the i-th egg in pick order with every later egg still in the carton
    (random kinds), and the i-th drop spot in the basket as the goal."""
    i = int(rng.integers(6))
    k = PICK_ORDER[i]
    kinds = list(rng.choice(["raw", "stone"], 6))
    kinds[k] = kind
    env.reset(kinds=kinds, present=PICK_ORDER[i:])
    env.set_target(k, drop=DROP_SPOTS[i])
    return k
