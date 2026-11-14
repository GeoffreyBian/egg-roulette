# Modelling notes

How the scene is built, why each MuJoCo setting is what it is, and how each choice was checked.
Everything here is reproduced by `python -m eggsim validate` (writes `results/physics.json`) and
`pytest -q`.

## Model construction

The SO-101 file from MuJoCo Menagerie stays unmodified on disk. `eggsim/scene.py` loads it into an
`mujoco.MjSpec`, edits the spec, adds the task, and compiles:

| Edit | How | Why |
|---|---|---|
| Gripper actuator | `set_to_motor()`, affine bias `-0.02 qdot`, ctrl range +-0.6 N m | Menagerie's jaw is a stiff position servo (kp 998). Its grip force is set by how far the jaw is told to close, which saturates the pads instantly. A torque actuator with our own P loop (kp 1.2 N m/rad, torque limit 0.35 N m) closed at the physics rate behaves like an STS3215 with its P coefficient turned down, and makes squeeze a smooth function of the command. |
| Jaw dynamics | armature 0.003, damping 0.04, frictionloss 0.004 | Lighter than the stock servo so the jaw doesn't hammer the egg on contact. |
| Pad contacts | every collision geom on the two jaw bodies: friction `[0.8, 0.012, 0.002]`, solref `[0.02, 1]`, solimp `[0.85, 0.95, 0.004, 0.5, 2]` | Jaw geoms carry `priority=1` in Menagerie, so their parameters win the contact mix. `condim=6` keeps torsional friction, sized to a fingertip-sized pad (12 mm), so the egg can't spin freely between two points. The soft solref spreads the impact over ~20 ms, like silicone. |
| Sensors | two `touch` sensors on box sites over each jaw's contact patch, `actuatorfrc` on the jaw (the servo-current proxy), `jointpos` on the jaw | Sites were placed from 3,300 measured jaw-egg contact points. |
| Task bodies | carton (pulp texture, 4 support bumps per cup), basket, six free-jointed ellipsoid eggs, four debris bodies per egg | Built with `add_body` / `add_geom`; textures are procedural (`scripts/make_textures.py`). |

Global options: `timestep 0.002`, `implicitfast`, `cone elliptic`, `impratio 10`.

## Per-episode model edits

Egg mass, inertia and radius are written into `MjModel` at every reset (radius jitter 0.5 mm, mass by
egg kind or domain randomisation), then `mj_setConst` recomputes derived quantities. Cracking is a
runtime topology change: the egg gets `gravcomp=1` and `contype=conaffinity=0` and is moved off stage,
and its yolk, white and shell pieces are switched on at the same pose and velocity. The broad phase
caches collision flags per body, so `body_contype` and `body_conaffinity` must be set as well as the geom
flags.

## Contact solver: why elliptic cones and impratio 10

MuJoCo contacts are soft: the solver computes forces that drive each constraint toward a spring-damper
reference acceleration set by solref/solimp, so a "static" grasp is really a very slow slide. How slow
depends on the friction cone and on `impratio` (the ratio of frictional to normal impedance).

The isolated pad rig (`validate.rig_model`) squeezes a 300 g egg between two pads with force
F = r m g / (2 mu) and measures how far it slides in 0.5 s:

| Solver | slip at r = 1.2 | slip at r = 2.0 |
|---|---|---|
| elliptic, impratio 10 (used) | 0.47 mm | 0.44 mm |
| pyramidal, impratio 10 | 1.48 mm | 1.33 mm |
| elliptic, impratio 1 | 4.40 mm | 4.17 mm |
| pyramidal, impratio 1 (MuJoCo default) | 14.9 mm | 13.0 mm |

With the defaults, an egg held at twice the Coulomb limit still creeps 13 mm in half a second. That
breaks the task: a stone egg would slide out of a correctly sized grip, and the learned policies would
pick up the solver's creep instead of the physics. With an elliptic cone and impratio 10, slip below the
limit is total (the egg falls) and above it is a 0.45 mm elastic settle that stops.

Two traps found while building the rig, both worth knowing:

1. **Pad mass sets contact stiffness.** solref is a time constant, so the force a given penetration
   produces scales with the constraint's effective mass. With 10 g pads the egg sank 7.6 mm into them; the
   real jaw's reflected inertia (armature / lever^2 ~ 0.8 kg) gives 1.5 mm in the rig and under 1 mm on
   the robot.
2. **Mocap bodies can't carry things by friction.** A mocap body is teleported each step with zero
   velocity, and MuJoCo's friction acts on velocities, so the egg stays behind when a mocap gripper
   lifts. The rig's carriage is a slide joint with prescribed position and velocity instead.

## Timestep and integrator

The rig lifts the egg 5 cm in 0.35 s along a minimum-jerk path (peak acceleration 2.36 m/s^2), and a
bisection finds the smallest squeeze ratio that carries it without sliding. Rigid-body theory says
r* = 1 + a_peak / g = 1.240.

| dt | 0.25 ms | 0.5 | 1 | 2 (used) | 4 | 8 |
|---|---|---|---|---|---|---|
| Euler | 1.216 | 1.216 | 1.216 | 1.216 | 1.217 | 1.218 |
| implicitfast | 1.216 | 1.216 | 1.216 | 1.216 | 1.217 | 1.218 |
| RK4 | 1.216 | 1.216 | 1.216 | 1.217 | 1.220 | 1.232 |

Converged to 3 decimals at 2 ms, within 2% of theory (the slip criterion allows 0.1 mm, which lets the
peak-acceleration instant through). Euler and implicitfast agree exactly here because the rig has no
velocity-dependent forces large enough to separate them; implicitfast is kept for the arm's damped servos.

## Squeeze, load and slip

`EggEnv.touch()` reads every active contact between the jaws and the target egg with `mj_contactForce`,
rotates it out of the contact frame, and reports

- squeeze: the smaller of the two jaws' summed normal forces (a one-sided push is not a pinch),
- load: the vertical component of the total jaw force on the egg,
- slip: the egg's speed relative to the gripper at the egg centre, from `mj_objectVelocity`.

Each is averaged over the 15 physics substeps of a 30 Hz control step, like a sensor integrating over its
sample period, and the crack check uses the same 33 ms average. The native touch sensors on the pad sites
report the same normal forces on every loaded substep checked (12,619, max error below 1e-6 N).

On the full robot the friction envelope is well below the pad coefficient. A fixed-squeeze controller
(force loop on the pad sensor, blind to load and slip) holds a squeeze F while a bisection over egg mass
finds the heaviest egg the SO-101 can hold:

| Test | heaviest egg vs squeeze | effective mu (2 pads) |
|---|---|---|
| Coulomb with pad mu | m g = 2 (0.8) F | 0.80 |
| lift 6 cm, hold still 1 s | linear | 0.53 |
| lift and carry to the basket | linear | 0.42 |

The relation stays linear, so it is still Coulomb friction, just with a smaller coefficient. Two things
are likely to account for the static gap, though they haven't been separated: the jaws touch the egg
through several small spheres and boxes on a curved shell, so contact normals tilt away from the grip axis
and only part of the measured squeeze presses along it; and the egg is held off its centre of mass, so its
weight also loads the torsional friction. The extra drop when carrying is the arm's accelerations. With
this envelope a stone egg needs about 3.5 N to carry and a raw egg cracks above 2.2 N, which is the
dilemma the task is built on.

## Speed

One control step is 15 physics steps plus an IK solve (damped least squares on `mj_jac`, 4 arm joints,
position plus an approach-axis term) and 15 contact readouts. On one Apple M4 core the env runs at about
380 control steps per second, 12.6x real time. Raw `mj_step` on this model runs at about 8,600 steps per
second; the Python-side contact loop is most of the overhead.

## Known gaps

- Eggs are rigid ellipsoids that crack at a force threshold. Real shells fail by stress concentration and
  take roughly 10x more force than 2.2 N; the threshold is scaled to the SO-101's 5 N gripper.
- The glove wearer is a scripted controller whose grip parameters (rise time, overshoot, catch-up
  squeezes) come from real human glove data. Arm paths are minimum-jerk, not recorded motion.
- The policy observes egg position directly (with 3 mm noise), not camera images.
