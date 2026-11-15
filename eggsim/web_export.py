"""Exports the scene, the policies and reference episodes for the in-browser simulator (web/).

The browser runs the same MuJoCo version (WebAssembly build of the version in requirements.txt) on the same
MJCF, so the physics is the physics in this repo, not a re-implementation. What changes for the web:

  * visual-only meshes are decimated (they have contype 0 and every body has an explicit <inertial>, so they
    don't touch the dynamics); the three jaw collision meshes are copied unchanged,
  * file textures are dropped from the MJCF (the renderer applies JPEG versions itself),
  * each policy is written as float16 weights plus its normalisation constants,
  * reference episodes record every action Python took, so web/test/parity.mjs can replay them and check
    that the browser port of env.py reaches the same states.
"""
from __future__ import annotations

import json
import pathlib
import re
import struct

import mujoco
import numpy as np

from .scene import ROBOT_XML, ROOT, TEXTURES, build_spec

OUT = ROOT / "web/assets"
POLICIES = ROOT / "data/policies"
VISUAL_GRID = 0.001                      # vertex-clustering cell for visual-only meshes, m


# ---------------------------------------------------------------- scene
def _write_stl(path, v, f):
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + struct.pack("<I", len(f)) + rec.tobytes())


def _read_stl(path):
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw[84:84 + 50 * n], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    v, inv = np.unique(np.round(tri, 7), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


def _cluster(v, f, h):
    """Vertex-clustering decimation: merge vertices on an h grid, drop collapsed and duplicate faces."""
    _, inv = np.unique(np.round(v / h).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    vs = np.zeros((inv.max() + 1, 3))
    np.add.at(vs, inv, v)
    vs /= np.bincount(inv)[:, None]
    f2 = inv[f]
    f2 = f2[(f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])]
    _, first = np.unique(np.sort(f2, 1), axis=0, return_index=True)
    return vs, f2[np.sort(first)]


def export_scene():
    spec = build_spec(6, render=True)
    m = spec.compile()
    collision = {m.mesh(m.geom_dataid[g]).name for g in range(m.ngeom)
                 if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]}
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    src = ROBOT_XML.parent / "assets"
    for mesh in spec.meshes:
        f = pathlib.Path(mesh.file).name
        if mesh.name in collision:
            (OUT / "meshes" / f).write_bytes((src / f).read_bytes())
        else:
            v, faces = _read_stl(src / f)
            _write_stl(OUT / "meshes" / f, *_cluster(v, faces, VISUAL_GRID))
    xml = spec.to_xml()
    xml = xml.replace('meshdir="assets"', 'meshdir="meshes"')
    xml = re.sub(r'\s*<texture type="2d"[^>]*file="[^"]*"/>', "", xml)
    xml = re.sub(r' texture="(wood|pulp)"', "", xml)
    (OUT / "scene.xml").write_text(xml)
    # renderer textures
    from PIL import Image
    for name in ("wood", "pulp"):
        Image.open(TEXTURES / f"{name}.png").convert("RGB").save(OUT / f"{name}.jpg", quality=82)
    print("scene:", len(xml), "bytes,", sum(p.stat().st_size for p in (OUT / "meshes").iterdir()) // 1024, "KB meshes")


# ---------------------------------------------------------------- policies
def _representative_seed(cond, n):
    """The seed whose success rate is closest to the median over seeds, so the demo shows a typical policy."""
    rows = [r for r in json.load(open(ROOT / "results/gain.json")) if r["cond"] == cond and r["n_robot"] == n]
    med = float(np.median([r["success"] for r in rows]))
    best = min(rows, key=lambda r: (abs(r["success"] - med), r["seed"]))
    return best["seed"], best["success"], [r["success"] for r in sorted(rows, key=lambda r: r["seed"])]


def _export_policy(src, name, **meta):
    import torch
    d = torch.load(src, weights_only=False)
    sd = d["model"]
    # trunk: (Linear, LayerNorm, GELU, Dropout) x 2 + (Linear, LayerNorm, GELU); then the action head
    keys = [f"{a}.{b}" for lin, ln in (("trunk.0", "trunk.1"), ("trunk.4", "trunk.5"), ("trunk.8", "trunk.9"))
            for a, b in ((lin, "weight"), (lin, "bias"), (ln, "weight"), (ln, "bias"))] + ["act.weight", "act.bias"]
    np.concatenate([sd[k].numpy().ravel() for k in keys]).astype(np.float16).tofile(OUT / "policies" / f"{name}.bin")
    cfg = d["cfg"]
    return dict(name=name, **meta, touch=cfg["touch"], hist=cfg["hist"], chunk=cfg["chunk"],
                touch_mask=list(cfg["touch_mask"]), proxy=cfg["proxy"],
                shapes=[list(sd[k].shape) for k in keys],
                mx=d["mx"].tolist(), sx=d["sx"].tolist(), my=d["my"].tolist(), sy=d["sy"].tolist())


def export_policies():
    (OUT / "policies").mkdir(parents=True, exist_ok=True)
    index = []
    for cond in "ABCD":
        for n in (10, 25, 50, 100):
            seed, succ, all_succ = _representative_seed(cond, n)
            index.append(_export_policy(POLICIES / f"{cond}_n{n}_s{seed}.pt", f"{cond}_n{n}", cond=cond, n_robot=n,
                                        seed=seed, success=succ, success_seeds=all_succ))
    for vid, name in VARIANTS.items():
        seed, succ, all_succ = _ablation_seed(name)
        src = POLICIES / f"var_{vid}_s{seed}.pt"
        if src.exists():
            index.append(_export_policy(src, f"var_{vid}", cond="C", n_robot=25, variant=vid, label=name[2:],
                                        seed=seed, success=succ, success_seeds=all_succ))
    (OUT / "policies/index.json").write_text(json.dumps(index))
    print("policies:", len(index))


# ---------------------------------------------------------------- training variants (from the ablations)
VARIANTS = {   # web id -> ablation name in experiments.ABLATIONS
    "no_chunks": "C no action chunks",
    "human50": "C 50 human demos",
    "lag300": "C touch lags video 300 ms",
    "pressure_only": "C grip pressure only",
    "servo": "C servo-current estimate",
    "varied": "C human demos on varied eggs",
}


def _ablation_seed(name):
    rows = [r for r in json.load(open(ROOT / "results/ablations.json")) if r["name"] == name]
    med = float(np.median([r["success"] for r in rows]))
    best = min(rows, key=lambda r: (abs(r["success"] - med), r["seed"]))
    return best["seed"], best["success"], [r["success"] for r in sorted(rows, key=lambda r: r["seed"])]


def _train_variant(vid):
    from .experiments import ABLATIONS
    from .policy import config, save, train
    name = VARIANTS[vid]
    seed = _ablation_seed(name)[0]
    path = POLICIES / f"var_{vid}_s{seed}.pt"
    if not path.exists():
        save(train(config(**ABLATIONS[name]), seed), path)
    return vid


def train_variants():
    """Retrains the median seed of each ablation the web demo offers (the ablation sweep doesn't keep policies)."""
    import multiprocessing as mp
    with mp.Pool(len(VARIANTS)) as pool:
        for vid in pool.imap_unordered(_train_variant, VARIANTS):
            print("trained", vid, flush=True)


# ---------------------------------------------------------------- reference episodes for the parity test
def export_reference(n_eps=6):
    """Python episodes with the float16 weights the browser uses, recorded action by action."""
    import torch
    from .env import EggEnv, setup_episode
    from .policy import Runner, load_policy, policy_episode
    eps = []
    for name in ("C_n25", "A_n25"):
        meta = next(p for p in json.load(open(OUT / "policies/index.json")) if p["name"] == name)
        pol = load_policy(POLICIES / f"{name}_s{meta['seed']}.pt")
        with torch.no_grad():
            for p in pol["model"].parameters():
                p.copy_(p.half().float())
        env = EggEnv(seed=777)
        rng = np.random.default_rng(31)
        runner = Runner(pol)
        for i in range(n_eps):
            kind = ("raw", "stone")[i % 2]
            k = setup_episode(env, rng, kind)
            init = dict(qpos=env.d.qpos.tolist(), qvel=env.d.qvel.tolist(), ctrl=env.d.ctrl.tolist(),
                        warmstart=env.d.qacc_warmstart.tolist(), egg_r=list(env.egg_r), kinds=[e.kind for e in env.eggs],
                        target=k, drop=env.drop.tolist(), ee_target=env.ee_target.tolist())
            obs_log, act_log, grip_log, ee_log = [], [], [], []
            act = runner.act

            def logged(o):                      # the exact policy inputs, vision noise included
                obs_log.append(o.tolist())
                return act(o)
            runner.act = logged

            def rec(o, a, info):
                act_log.append(a.tolist()); grip_log.append(info.grip_n); ee_log.append(env.ee().tolist())
            out, t = policy_episode(env, runner, record=rec)
            runner.act = act
            eps.append(dict(policy=name, kind=kind, init=init, obs=obs_log, actions=act_log, grip=grip_log,
                            ee=ee_log, outcome=out,
                            steps=t, final_ee=env.ee().tolist(), final_egg=env.egg_pos(k).tolist()))
            print(name, kind, out, t)
    (ROOT / "web/test").mkdir(parents=True, exist_ok=True)
    (ROOT / "web/test/reference.json").write_text(json.dumps(eps))


def main(args=None):
    args = args or ["variants", "scene", "policies", "reference"]
    if "variants" in args:
        train_variants()
    if "scene" in args:
        export_scene()
    if "policies" in args:
        export_policies()
    if "reference" in args:
        export_reference()


if __name__ == "__main__":
    main()
