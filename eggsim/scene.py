"""Builds the egg-roulette MuJoCo model with the MjSpec API.

The SO-101 comes from MuJoCo Menagerie unchanged on disk. Everything task-specific is applied to the spec
before compiling, so the robot file stays a clean upstream copy:

  * the gripper's position servo is replaced by a torque-limited current loop (we close the position loop
    ourselves at the physics rate, like lowering the STS3215's P coefficient),
  * the jaw collision geoms get silicone-pad contact parameters (friction, torsional friction, soft solref),
  * native `touch` sensors are attached to the contact patch of each jaw, plus an `actuatorfrc` sensor that
    plays the role of the servo's current reading,
  * the carton, basket and eggs are added as bodies, with crack debris parked off stage.

Every physical constant tagged (A) is an assumption chosen to make the look-alike problem real at SO-101
scale, not a measurement. See docs/physics.md.
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROBOT_XML = ROOT / "assets/so101/so101.xml"
TEXTURES = ROOT / "assets/textures"

# ---------------------------------------------------------------- task geometry and physics constants
EGG_R, EGG_H = 0.021, 0.026          # ellipsoid semi-axes: 42 mm wide, 52 mm tall
EGG_R_SD = 0.0005                    # egg-to-egg radius spread, m (A)
MU = 0.80                            # silicone pad on eggshell (A)
PAD_TORSION = 0.012                  # torsional friction length of a finger-sized pad, m (A)
PAD_SOLREF = (0.02, 1.0)             # soft pad: 20 ms time constant, critically damped
PAD_SOLIMP = (0.85, 0.95, 0.004, 0.5, 2)
EGG_SOLREF = (0.005, 1.0)
EGG_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2)

SIM_DT = 0.002
CTRL_HZ = 30

CARTON_ORIGIN = np.array([0.155, 0.04])
SLOT_PITCH = 0.064
SLOTS = [CARTON_ORIGIN + SLOT_PITCH * np.array([i, j]) for j in range(3) for i in range(2)]  # 2 x 3
CARTON_FLOOR = 0.006
SUPPORT_R = 0.0175                   # support bumps sit this far from the slot centre, on the diagonals
BASKET_C = np.array([0.17, -0.13])
BASKET_HALF = 0.075
BASKET_WALL_H = 0.035

# contact patches measured from 3,300 jaw-egg contacts during scripted grasps (see docs/physics.md)
FIXED_PAD_SITE = dict(pos=[-0.0088, 0.0, -0.0945], size=[0.0045, 0.0065, 0.0125])     # gripper body frame
MOVING_PAD_SITE = dict(pos=[-0.0104, -0.066, 0.0194], size=[0.0045, 0.0185, 0.0075])  # moving jaw frame

PARK = np.array([5.0, 5.0, 0.03])    # where unused bodies wait, far outside the workspace
DEBRIS = ("yolk", "white", "shellA", "shellB")


def _rgba(*c):
    return list(c) + ([1.0] if len(c) == 3 else [])


def build_spec(n_eggs: int = 6, render: bool = True) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.modelname = "egg_roulette"
    opt = spec.option
    opt.timestep = SIM_DT
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10                 # stiff friction relative to normal: no creep under a static hold

    _gripper_as_current_loop(spec)
    _pad_contacts(spec)
    _sensors(spec)
    if render:
        _visuals(spec)
    _table_carton_basket(spec)
    _eggs(spec, n_eggs)
    return spec


def build_model(n_eggs: int = 6, render: bool = True) -> mujoco.MjModel:
    return build_spec(n_eggs, render).compile()


# ---------------------------------------------------------------- robot edits
def _gripper_as_current_loop(spec):
    act = spec.actuator("gripper")
    act.set_to_motor()               # ctrl is joint torque (N m); the P loop runs in env.step at 500 Hz
    act.gear[0] = 1.0
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm[:3] = [0.0, 0.0, -0.02]      # back-EMF style velocity damping
    act.ctrlrange = [-0.6, 0.6]
    act.ctrllimited = 1
    act.forcelimited = 0
    jnt = spec.joint("gripper")
    jnt.damping[0], jnt.frictionloss = 0.04, 0.004
    jnt.armature = 0.003             # jaw inertia seen through the gearbox, lighter than a stock STS3215 (A)


def _pad_contacts(spec):
    for body in ("gripper", "moving_jaw_so101_v1"):
        for g in spec.body(body).geoms:
            # jaw geoms win contact priority, so their parameters set the egg contact
            g.friction = [MU, PAD_TORSION, 0.002]
            g.solref = list(PAD_SOLREF)
            g.solimp = list(PAD_SOLIMP)


def _sensors(spec):
    sites = {}
    for body, cfg, name in (("gripper", FIXED_PAD_SITE, "pad_fixed"),
                            ("moving_jaw_so101_v1", MOVING_PAD_SITE, "pad_moving")):
        s = spec.body(body).add_site(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=cfg["pos"], size=cfg["size"],
                                     rgba=[0.1, 0.8, 0.9, 0.35], group=4)
        sites[name] = s
        spec.add_sensor(name=f"touch_{name[4:]}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name)
    spec.add_sensor(name="gripper_torque", type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
                    objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname="gripper")
    spec.add_sensor(name="gripper_pos", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                    objtype=mujoco.mjtObj.mjOBJ_JOINT, objname="gripper")


# ---------------------------------------------------------------- scene
def _visuals(spec):
    v = spec.visual
    v.headlight.diffuse = [0.45, 0.45, 0.45]
    v.headlight.ambient = [0.32, 0.32, 0.34]
    v.headlight.specular = [0.05, 0.05, 0.05]
    v.rgba.haze = [0.14, 0.16, 0.2, 1]
    v.global_.offwidth, v.global_.offheight = 1920, 1080
    v.quality.shadowsize = 8192
    v.quality.offsamples = 8
    v.map.znear = 0.005
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.30, 0.34, 0.40], rgb2=[0.06, 0.07, 0.09], width=512, height=3072)


def _material(spec, name, rgba, texture=None, **kw):
    m = spec.add_material(name=name, rgba=rgba, **kw)
    if texture:
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture
    return m


def _table_carton_basket(spec):
    spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(TEXTURES / "wood.png"))
    spec.add_texture(name="pulp", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(TEXTURES / "pulp.png"))
    _material(spec, "table", [1, 1, 1, 1], "wood", texrepeat=[0.9, 0.9], texuniform=True, reflectance=0.03,
              specular=0.25, shininess=0.3)
    _material(spec, "carton", [1, 1, 1, 1], "pulp", texrepeat=[4, 4], texuniform=True, specular=0.05)
    _material(spec, "basket", _rgba(0.30, 0.19, 0.11), specular=0.15, shininess=0.2)
    _material(spec, "shell", _rgba(0.97, 0.91, 0.80), specular=0.3, shininess=0.5)
    _material(spec, "yolk", _rgba(1.0, 0.62, 0.05), specular=0.6, shininess=0.9)
    _material(spec, "white", [1.0, 1.0, 0.96, 0.85], specular=0.7, shininess=0.9)

    wb = spec.worldbody
    wb.add_light(pos=[0.35, 0.25, 1.1], dir=[-0.25, -0.15, -1], castshadow=True, diffuse=[0.62, 0.6, 0.56],
                 specular=[0.25, 0.25, 0.25])
    wb.add_light(pos=[-0.5, -0.6, 0.9], dir=[0.45, 0.5, -0.8], castshadow=False, diffuse=[0.22, 0.24, 0.28])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="table",
                friction=[0.8, 0.01, 0.001])

    carton = wb.add_body(name="carton")
    cx, cy = np.mean(SLOTS, axis=0)
    carton.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, CARTON_FLOOR / 2],
                    size=[0.068, 0.105, CARTON_FLOOR / 2], material="carton")
    w, h, t = 0.0285, 0.007, 0.0035
    for x, y in SLOTS:
        # visual cup walls, plus four support bumps that hold the egg upright like a real carton
        for pos, size in (([x + w, y], [t, w, h]), ([x - w, y], [t, w, h]),
                          ([x, y + w], [w, t, h]), ([x, y - w], [w, t, h])):
            carton.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos + [CARTON_FLOOR + h], size=size,
                            material="carton", contype=0, conaffinity=0)
        r = SUPPORT_R / np.sqrt(2)
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            carton.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[x + dx * r, y + dy * r, CARTON_FLOOR + 0.004],
                            size=[0.0055, 0, 0], material="carton", friction=[0.3, 0.01, 0.001])

    basket = wb.add_body(name="basket", pos=list(BASKET_C) + [0])
    b, wh, t = BASKET_HALF, BASKET_WALL_H / 2, 0.004
    basket.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.003], size=[b, b, 0.003], material="basket",
                    friction=[1.5, 0.05, 0.01])
    for pos, size in (([b, 0], [t, b, wh]), ([-b, 0], [t, b, wh]), ([0, b], [b, t, wh]), ([0, -b], [b, t, wh])):
        basket.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos + [wh], size=size, material="basket")

    wb.add_camera(name="front", pos=[0.62, -0.02, 0.36], xyaxes=[0.05, 1, 0, -0.45, 0.02, 0.89], fovy=50)
    wb.add_camera(name="side", pos=[0.24, -0.55, 0.30], xyaxes=[1, 0, 0, 0, 0.5, 0.87], fovy=48)
    wb.add_camera(name="close", pos=[0.46, 0.18, 0.20], xyaxes=[-0.3, 0.95, 0, -0.35, -0.11, 0.93], fovy=45)
    wb.add_camera(name="hero", pos=[0.50, -0.36, 0.30], xyaxes=[0.62, 0.78, 0, -0.42, 0.33, 0.84], fovy=42)


def _eggs(spec, n_eggs):
    wb = spec.worldbody
    E = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    for k in range(n_eggs):
        egg = wb.add_body(name=f"egg{k}", pos=[PARK[0] + k * 0.2, PARK[1], PARK[2]])
        egg.add_freejoint(name=f"egg{k}")
        egg.add_geom(name=f"egg{k}", type=E, size=[EGG_R, EGG_R, EGG_H], material="shell", mass=0.06,
                     friction=[MU, 0.01, 0.001], condim=4, solref=list(EGG_SOLREF), solimp=list(EGG_SOLIMP))
        # crack debris, weightless and non-colliding until the egg cracks
        parts = {"yolk": ([0.024, 0.024, 0.007], "yolk", 0.02), "white": ([0.052, 0.040, 0.003], "white", 0.02),
                 "shellA": ([0.017, 0.013, 0.006], "shell", 0.006), "shellB": ([0.014, 0.011, 0.005], "shell", 0.006)}
        for j, name in enumerate(DEBRIS):
            size, mat, mass = parts[name]
            body = wb.add_body(name=f"{name}{k}", pos=[PARK[0] + k * 0.2, PARK[1] + 1 + 0.3 * j, 0.02], gravcomp=1)
            body.add_freejoint()
            body.add_geom(type=E, size=size, material=mat, mass=mass, contype=0, conaffinity=0,
                          friction=[2, 0.1, 0.1], solref=[0.02, 1.5])
