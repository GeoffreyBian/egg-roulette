"""Ray-cast cameras: what the pixel policies see.

One ray per pixel through MuJoCo's own ray caster (mj_multiRay, with surface normals) against the collision
geometry, shaded with a flat colour per object and one directional light. The browser demo calls the same
function of the same engine on the same model (web/sim/camera.js), so a policy sees the same pixels in
training and in the browser; web/test/parity.mjs checks it.

Why not MuJoCo's OpenGL renderer: the WebAssembly build has no renderer, and matching a three.js render to it
pixel for pixel isn't possible. Rays against collision geometry are also cheap: about 4 ms for a 64x64 view.
The robot shows up as its collision boxes and jaw hulls, which is what a coarse depth-and-colour camera would
resolve anyway. Raw and stone eggs share a geometry and a colour, so pixels can't tell them apart.
"""
from __future__ import annotations

import mujoco
import numpy as np

RES = 64
CAMERAS = ("workspace", "wrist_cam")
GROUPS = np.array([1, 1, 0, 1, 1, 0], np.uint8)     # scene + collision geoms; visual meshes (group 2) skipped
LIGHT = np.array([0.35, 0.25, 1.0]) / np.linalg.norm([0.35, 0.25, 1.0])
AMBIENT, DIFFUSE = 0.38, 0.62
BACKGROUND = np.array([0.16, 0.18, 0.21])
CUTOFF = 3.0

# flat colours by what a geom belongs to (sRGB, 0-1)
COLOURS = {
    "table": (0.74, 0.58, 0.42), "carton": (0.80, 0.74, 0.62), "basket": (0.34, 0.22, 0.13),
    "egg": (0.96, 0.91, 0.82), "yolk": (1.0, 0.62, 0.05), "white": (0.97, 0.97, 0.92), "shell": (0.96, 0.91, 0.82),
    "robot": (0.98, 0.80, 0.14), "jaw": (0.92, 0.55, 0.10),
}


def geom_colours(m: mujoco.MjModel) -> np.ndarray:
    """(ngeom, 3) colour per geom, by body name."""
    out = np.zeros((m.ngeom, 3))
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if g == 0 or m.geom(g).name == "table":
            key = "table"
        elif body in ("carton", "basket"):
            key = body
        elif body.startswith("egg"):
            key = "egg"
        elif body.rstrip("0123456789") in ("yolk", "white", "shellA", "shellB"):
            key = {"shellA": "shell", "shellB": "shell"}.get(body.rstrip("0123456789"), body.rstrip("0123456789"))
        elif body in ("gripper", "moving_jaw_so101_v1"):
            key = "jaw"
        else:
            key = "robot"
        out[g] = COLOURS[key]
    return out


def ray_dirs(fovy_deg: float, res: int = RES) -> np.ndarray:
    """Unit ray directions in the camera frame (x right, y up, looking down -z), row-major from the top-left."""
    t = np.tan(np.deg2rad(fovy_deg) / 2)
    u = ((np.arange(res) + 0.5) / res * 2 - 1) * t
    X, Y = np.meshgrid(u, -u)
    v = np.stack([X, Y, -np.ones_like(X)], -1).reshape(-1, 3)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class RayCamera:
    def __init__(self, m: mujoco.MjModel, name: str, res: int = RES):
        self.m, self.res = m, res
        self.cam = m.camera(name).id
        self.exclude = int(m.cam_bodyid[self.cam]) if m.cam_bodyid[self.cam] > 0 else -1   # the camera's own lens
        self.local = ray_dirs(float(m.cam_fovy[self.cam]), res)
        self.colours = geom_colours(m)
        n = res * res
        self.gid = np.zeros(n, np.int32)
        self.dist = np.zeros(n)
        self.normal = np.zeros(3 * n)

    def render(self, d: mujoco.MjData) -> np.ndarray:
        """(res, res, 3) uint8. Needs the position stage current for d.qpos (see Cameras.render)."""
        m, n = self.m, self.res * self.res
        R = d.cam_xmat[self.cam].reshape(3, 3)
        vec = self.local @ R.T
        mujoco.mj_multiRay(m, d, d.cam_xpos[self.cam], vec.ravel(), GROUPS, 1, self.exclude, self.gid, self.dist, self.normal,
                           n, CUTOFF)
        return shade(self.gid, self.normal.reshape(-1, 3), vec, self.colours).reshape(self.res, self.res, 3)


def shade(gid, normal, vec, colours):
    hit = gid >= 0
    nrm = np.where((np.sum(normal * vec, 1) > 0)[:, None], -normal, normal)     # face the camera
    lam = np.clip(nrm @ LIGHT, 0, 1)
    rgb = np.where(hit[:, None], colours[np.maximum(gid, 0)] * (AMBIENT + DIFFUSE * lam)[:, None], BACKGROUND)
    return np.floor(np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


class Cameras:
    """Both policy cameras. render(d) -> (2, res, res, 3) uint8."""

    def __init__(self, m: mujoco.MjModel, res: int = RES):
        self.m = m
        self.cams = [RayCamera(m, c, res) for c in CAMERAS]

    def render(self, d: mujoco.MjData) -> np.ndarray:
        # mj_multiRay culls with the bounding boxes that collision detection computes, so run the whole position
        # stage for the current qpos (mj_kinematics alone leaves them stale). Physics is unaffected: mj_step
        # recomputes everything from qpos and qvel.
        mujoco.mj_fwdPosition(self.m, d)
        return np.stack([c.render(d) for c in self.cams])
