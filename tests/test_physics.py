"""Fast checks of the model and the contact physics the experiments depend on. Run: pytest -q"""
import mujoco
import numpy as np
import pytest

from eggsim import validate
from eggsim.demos import HumanGloveDemo, TeleopDemo, run_episode
from eggsim.env import CRACK_N, EggEnv, StepInfo, setup_episode
from eggsim.scene import MU, build_model


def test_model_compiles_with_sensors_and_current_loop():
    m = build_model()
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == 10
    for name in ("touch_fixed", "touch_moving", "gripper_torque", "gripper_pos"):
        assert m.sensor(name).id >= 0
    a = m.actuator("gripper").id
    assert m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED and m.actuator_gainprm[a, 0] == 1.0
    assert m.actuator_biasprm[a, 2] == pytest.approx(-0.02)


@pytest.mark.parametrize("ratio,held", [(0.8, False), (1.3, True)])
def test_rig_holds_above_coulomb_limit_only(ratio, held):
    m = validate.rig_model(mass=0.3)
    d = mujoco.MjData(m)
    validate._hold(m, d, ratio * 0.3 * 9.81 / (2 * MU))
    z0 = d.xpos[m.body("egg").id][2]
    for _ in range(250):
        validate._step(m, d)
    slip_mm = (z0 - d.xpos[m.body("egg").id][2]) * 1000
    assert (slip_mm < 1.0) == held


def test_touch_sensors_match_contact_forces():
    env = EggEnv(seed=1)
    rng = np.random.default_rng(1)
    setup_episode(env, rng, "stone")
    demo, info = HumanGloveDemo(env, rng), StepInfo()
    while demo.phase != "carry":
        _, info = env.step(demo.act(info))
    ti = env.touch()
    fixed, moving = env.native_touch()
    assert min(fixed, moving) == pytest.approx(ti.grip_n, rel=1e-6, abs=1e-6)


def test_raw_egg_cracks_and_stone_does_not():
    for kind, cracks in (("raw", True), ("stone", False)):
        env = EggEnv(seed=2)
        rng = np.random.default_rng(2)
        setup_episode(env, rng, kind)
        demo = TeleopDemo(env, rng)
        demo.depth = 0.32                   # a very firm squeeze: about 5 N at the pads
        _, out = run_episode(env, demo, max_steps=330)
        assert (out == "cracked") == cracks
        assert env.eggs[env.target].peak_grip > CRACK_N


def test_simulation_is_deterministic():
    traj = []
    for _ in range(2):
        env = EggEnv(seed=5)
        rng = np.random.default_rng(5)
        setup_episode(env, rng, "raw")
        demo, info = HumanGloveDemo(env, rng), StepInfo()
        for _ in range(60):
            _, info = env.step(demo.act(info))
        traj.append(env.d.qpos.copy())
    np.testing.assert_array_equal(traj[0], traj[1])
