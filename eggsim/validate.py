"""Physics checks: does the contact model do what the task depends on?

The whole experiment rests on one fact: an egg held by two pads slips when the load exceeds the friction
cone, load > 2 mu F. MuJoCo's contacts are soft (constraint forces come from a spring-damper reference
acceleration, solref/solimp), so that fact holds only for some solver settings. These checks measure it.

  1. friction cone   isolated two-pad rig, egg held at squeeze F = r * m g / (2 mu); slip speed vs r
                     for elliptic/pyramidal cones and impratio 1/10
  2. convergence     the rig lifts the egg with a fast min-jerk move; slip distance vs timestep for
                     Euler, implicitfast and RK4, plus the wall-clock cost of each
  3. robot cone      full SO-101: lift a stone egg, then slowly open the jaw; every substep's
                     (squeeze, load, slip) against the 2 mu F line
  4. touch sensors   MuJoCo touch sensors on the pad sites vs the contact-force ground truth
  5. speed           env steps per second, real-time factor

Writes results/physics.json.
"""
from __future__ import annotations

import json
import time

import mujoco
import numpy as np

from .scene import EGG_H, EGG_R, EGG_SOLIMP, EGG_SOLREF, MU, PAD_SOLIMP, PAD_SOLREF, PAD_TORSION, ROOT

G = 9.81
PAD_MASS = 0.8      # the SO-101 jaw's reflected inertia at the pad: armature / lever^2 = 0.003 / 0.06^2 kg


# ---------------------------------------------------------------- isolated rig
def rig_model(mass=0.3, cone="elliptic", impratio=10.0, dt=0.002, integrator="implicitfast",
              solref=PAD_SOLREF) -> mujoco.MjModel:
    """Two pads on slide joints squeeze an egg. A carriage on a vertical slide carries both pads, so it can lift.

    The carriage is a prescribed joint, not a mocap body: mocap bodies teleport with zero velocity, and
    MuJoCo's friction acts on velocities, so a mocap gripper can't carry anything by friction."""
    s = mujoco.MjSpec()
    s.option.timestep = dt
    s.option.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[cone]
    s.option.impratio = impratio
    s.option.integrator = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                           "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}[integrator]
    # heavy and gravity-compensated, so contact reactions can't push it off its prescribed path
    car = s.worldbody.add_body(name="carriage", pos=[0, 0, 0.1], gravcomp=1)
    car.add_joint(name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    car.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.01, 0.003], pos=[0, 0, 0.02], mass=500.0,
                 contype=0, conaffinity=0)
    for side in (-1, 1):
        pad = car.add_body(name=f"pad{side:+d}", pos=[side * (EGG_R + 0.004), 0, 0], gravcomp=1)
        pad.add_joint(name=f"slide{side:+d}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], damping=0.5)
        pad.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.01, 0.012], mass=PAD_MASS, priority=1, condim=6,
                     friction=[MU, PAD_TORSION, 0.002], solref=list(solref), solimp=list(PAD_SOLIMP))
        a = s.add_actuator(name=f"push{side:+d}", target=f"slide{side:+d}",
                           trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_motor()
        a.ctrllimited = 0
    egg = s.worldbody.add_body(name="egg", pos=[0, 0, 0.1])
    egg.add_freejoint()
    egg.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[EGG_R, EGG_R, EGG_H], mass=mass, condim=4,
                 friction=[MU, 0.01, 0.001], solref=list(EGG_SOLREF), solimp=list(EGG_SOLIMP))
    return s.compile()


def _step(m, d, z=0.0, vz=0.0):
    """One step with the carriage on its prescribed path."""
    d.qpos[0], d.qvel[0] = z, vz
    mujoco.mj_step(m, d)


def _hold(m, d, F, t_settle=0.3):
    """Squeeze with F under zero gravity until the contact settles, then switch gravity on."""
    m.opt.gravity[:] = 0
    d.ctrl[:] = [F, -F]                 # pad -1 pushes +x, pad +1 pushes -x
    for _ in range(int(t_settle / m.opt.timestep)):
        _step(m, d)
    m.opt.gravity[:] = [0, 0, -G]


def friction_cone(ratios=np.linspace(0.5, 2.5, 41), mass=0.3, t_hold=0.5):
    configs = {"elliptic, impratio 10 (used)": ("elliptic", 10), "elliptic, impratio 1": ("elliptic", 1),
               "pyramidal, impratio 10": ("pyramidal", 10), "pyramidal, impratio 1 (default)": ("pyramidal", 1)}
    f_star = mass * G / (2 * MU)
    out = {"ratios": list(map(float, ratios)), "f_star": f_star, "mass": mass, "curves": {}}
    for name, (cone, ir) in configs.items():
        m = rig_model(mass, cone, ir)
        d = mujoco.MjData(m)
        egg = m.body("egg").id
        v = []
        for r in ratios:
            mujoco.mj_resetData(m, d)
            _hold(m, d, r * f_star)
            z0 = d.xpos[egg][2]
            for _ in range(int(t_hold / m.opt.timestep)):
                _step(m, d)
            v.append(float(min(z0 - d.xpos[egg][2], 0.03) * 1000))     # slip in t_hold, mm; 30 = dropped
        out["curves"][name] = v
        print(f"  cone {name:32s} slip in {t_hold} s at r=0.9: {np.interp(0.9, ratios, v):5.2f} mm, "
              f"r=1.2: {np.interp(1.2, ratios, v):5.2f} mm, r=2: {np.interp(2, ratios, v):5.2f} mm", flush=True)
    return out


def _lift_slip(m, ratio=1.3, mass=0.3, lift=0.05, T=0.35):
    """Lift the carriage `lift` m in T s (min-jerk, peak accel ~5.8 lift/T^2), return slip distance in mm."""
    d = mujoco.MjData(m)
    _hold(m, d, ratio * mass * G / (2 * MU))
    egg = m.body("egg").id
    rel0 = d.xpos[egg][2] - d.qpos[0]
    n = int((T + 0.3) / m.opt.timestep)
    t0 = time.perf_counter()
    for i in range(n):
        s = min(1.0, i * m.opt.timestep / T)
        z = lift * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
        vz = lift / T * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) if s < 1 else 0.0
        _step(m, d, z, vz)
    wall = time.perf_counter() - t0
    return float((rel0 - (d.xpos[egg][2] - d.qpos[0])) * 1000), wall / (n * m.opt.timestep)


def _static_sag(m, ratio, t=0.65):
    """Elastic sag of a still hold at the same squeeze: soft contacts give a little under load, then stop."""
    d = mujoco.MjData(m)
    _hold(m, d, ratio * m.body_mass[m.body("egg").id] * G / (2 * MU))
    egg = m.body("egg").id
    z0 = d.xpos[egg][2]
    for _ in range(int(t / m.opt.timestep)):
        _step(m, d)
    return float((z0 - d.xpos[egg][2]) * 1000)


def critical_ratio(m, lo=0.9, hi=2.0, iters=12, tol_mm=0.1):
    """Smallest squeeze ratio r (in units of the static limit m g / 2 mu) that carries the egg through the lift
    without sliding: slip after the lift, beyond the elastic sag of a still hold at the same r, under 0.1 mm."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        slip = _lift_slip(m, ratio=mid)[0] - _static_sag(m, mid)
        lo, hi = (lo, mid) if np.isfinite(slip) and slip < tol_mm else (mid, hi)
    return hi


def convergence(dts=(0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008), lift=0.05, T=0.35):
    """Critical squeeze ratio during a min-jerk lift vs timestep and integrator. Rigid-body theory: the
    pads must carry m (g + a_peak), so r* = 1 + a_peak / g with a_peak = 5.774 lift / T^2."""
    theory = 1 + 5.774 * lift / T ** 2 / G
    out = {"dts": list(dts), "theory": theory, "r_crit": {}, "wall_per_sim_s": {}}
    for integ in ("euler", "implicitfast", "rk4"):
        rc, wc = [], []
        for dt in dts:
            m = rig_model(dt=dt, integrator=integ)
            rc.append(critical_ratio(m))
            wc.append(_lift_slip(m, ratio=1.5)[1])
        out["r_crit"][integ], out["wall_per_sim_s"][integ] = rc, wc
        print(f"  {integ:12s} r* by dt: {np.round(rc, 3)}  (theory {theory:.3f})", flush=True)
    return out


# ---------------------------------------------------------------- full robot
def robot_cone(forces=(0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), iters=9, seed=3):
    """Full SO-101: hold a fixed squeeze F (force loop on the pad sensor, no load tracking, no slip reflex) and
    bisect for the heaviest egg it can (a) lift 6 cm and hold still for 1 s, (b) lift and carry to the basket.
    Coulomb with two pads predicts m g = 2 mu F."""
    from .demos import HumanGloveDemo
    from .env import EggEnv, StepInfo, setup_episode

    class FixedSqueeze(HumanGloveDemo):
        def __init__(self, env, rng, force):
            super().__init__(env, rng, retarget_noise=0.0)
            self.F_START, self.SAFETY, self.f_target = force, 0.0, force
            self.rise, self.overshoot = 0.3, 1.0

        def grip(self, info):
            info = StepInfo(grip_n=info.grip_n, shear_n=0.0, slip=0.0)   # blind to load and slip
            return super().grip(info)

    def holds(force, mass, carry):
        env = EggEnv(seed=seed, vision_noise=0.0)
        rng = np.random.default_rng(seed)
        setup_episode(env, rng, f"m{mass:.4f}_cinf")
        demo, info, grips = FixedSqueeze(env, rng, force), StepInfo(), []
        while demo.phase != ("lower" if carry else "carry") and not demo.done:
            _, info = env.step(demo.act(info))
        if not carry:                     # hold still at the top of the lift for 1 s
            for _ in range(30):
                _, info = env.step(np.r_[0, 0, 0, demo.grip(info)])
                grips.append(info.grip_n)
        elif env.egg_status() == "held":
            grips.append(info.grip_n)
        return env.egg_status() == "held", float(np.mean(grips)) if grips else force

    out = {}
    for mode in ("still", "carry"):
        pts = []
        for F in forces:
            lo, hi, g_meas = 0.0, 0.8, F
            for _ in range(iters):
                mid = (lo + hi) / 2
                ok, g = holds(F, mid, mode == "carry")
                if ok:
                    lo, g_meas = mid, g
                else:
                    hi = mid
            pts.append(dict(force_cmd=F, squeeze=g_meas, max_mass=lo, max_weight=lo * G))
            print(f"  robot {mode}: squeeze {g_meas:4.2f} N holds up to {lo * 1000:5.0f} g  "
                  f"(Coulomb {2 * MU * g_meas / G * 1000:5.0f} g)", flush=True)
        sq = np.array([p["squeeze"] for p in pts])
        wt = np.array([p["max_weight"] for p in pts])
        mu = float((sq @ wt) / (sq @ sq) / 2)            # least-squares slope through the origin, / 2 pads
        print(f"  robot {mode}: mu_eff = {mu:.3f} (pad mu {MU})", flush=True)
        out[mode] = dict(points=pts, mu_eff=mu)
    return out


def touch_check(n_eps=6, seed=11):
    """Native touch sensors (normal force summed over contacts inside each pad site) vs mj_contactForce."""
    from .demos import HumanGloveDemo, run_episode
    from .env import EggEnv, setup_episode
    env = EggEnv(seed=seed)
    rng = np.random.default_rng(seed)
    gt, sens = [], []
    orig_step = env.step

    def step(a, on_substep=None):
        def cb(ti):
            m, d = env.m, env.d
            f6 = np.zeros(6)
            per = [0.0, 0.0]
            eggs = set(env.egg_geom)
            for i in range(d.ncon):
                c = d.contact[i]
                for gid, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if gid in env.jaw_geoms and other in eggs:
                        mujoco.mj_contactForce(m, d, i, f6)
                        per[int(m.geom_bodyid[gid] == env.moving_jaw)] += abs(f6[0])
            gt.append(per)
            sens.append(env.native_touch())
        return orig_step(a, on_substep=cb)
    env.step = step
    for i in range(n_eps):
        setup_episode(env, rng, "stone" if i % 2 else "raw")
        run_episode(env, HumanGloveDemo(env, rng), max_steps=330)
    gt, sens = np.array(gt), np.array(sens)
    on = gt.sum(1) > 0.05
    err = np.abs(sens[on] - gt[on])
    print(f"  touch sensors: {on.sum()} loaded substeps, median |error| {np.median(err):.4f} N, "
          f"p99 {np.percentile(err, 99):.3f} N", flush=True)
    keep = np.flatnonzero(on)[:: max(1, on.sum() // 3000)]
    return dict(gt=gt[keep].tolist(), sensor=sens[keep].tolist(), median_abs_err=float(np.median(err)),
                p99_abs_err=float(np.percentile(err, 99)), n=int(on.sum()))


def speed(n=600):
    from .env import CTRL_HZ, EggEnv
    env = EggEnv(seed=0)
    env.reset()
    a = np.zeros(4)
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(a)
    wall = time.perf_counter() - t0
    m = env.m
    d = env.d
    t0 = time.perf_counter()
    for _ in range(5000):
        mujoco.mj_step(m, d)
    raw = 5000 / (time.perf_counter() - t0)
    out = dict(ctrl_steps_per_s=n / wall, realtime_factor=n / CTRL_HZ / wall, raw_physics_steps_per_s=raw,
               nbody=m.nbody, ngeom=m.ngeom, nv=m.nv, mujoco=mujoco.__version__)
    print(f"  speed: {out['ctrl_steps_per_s']:.0f} control steps/s ({out['realtime_factor']:.1f}x real time), "
          f"{raw:.0f} raw physics steps/s", flush=True)
    return out


def main():
    res = {}
    for name, fn in (("friction_cone", friction_cone), ("convergence", convergence), ("robot_cone", robot_cone),
                     ("touch_check", touch_check), ("speed", speed)):
        print(name, flush=True)
        res[name] = fn()
    (ROOT / "results/physics.json").write_text(json.dumps(res))
