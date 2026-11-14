"""Media: side-by-side video, README gif, hero still, filmstrip, and the traces behind figures/episode_anatomy.png.

Both robots get the same carton (same egg kinds, same seed). The carton is chosen, without rendering, as the
one of 12 random cartons whose basket counts sit closest to each policy's 60-egg test average, so the video
shows typical behaviour rather than a lucky run.

  python -m eggsim render            # everything
  python -m eggsim render anatomy    # just results/anatomy.json
"""
from __future__ import annotations

import json
import subprocess

import imageio.v2 as imageio
import mujoco
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from .env import CRACK_N, DROP_SPOTS, HOME, PICK_ORDER, EggEnv, flat_obs, setup_episode
from .experiments import POLICIES
from .policy import Runner, load_policy, policy_episode
from .scene import ROOT

MEDIA = ROOT / "media"
W, H, FPS = 960, 600, 30
SPEC = {"A": ("A", 25, 0), "C": ("C", 25, 0)}
TITLE = {"A": "Robot demos only", "C": "+ human demos with touch"}
RGB = {"A": (42, 120, 214), "C": (235, 104, 52)}
OUT_RGB = {"success": (12, 163, 12), "cracked": (208, 59, 59), "dropped": (236, 131, 90), "timeout": (250, 178, 25)}
OUT_TXT = {"success": "in the basket", "cracked": "cracked", "dropped": "dropped", "timeout": "couldn't lift it"}


def _font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family=["Helvetica Neue", "Arial", "DejaVu Sans"],
                                                             weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _pol(c):
    cond, n, s = SPEC[c]
    return load_policy(POLICIES / f"{cond}_n{n}_s{s}.pt")


# ---------------------------------------------------------------- HUD
def _chip(dr, xy, text, font, fill=(16, 18, 22, 210), ink=(255, 255, 255), pad=(12, 7), dot=None):
    x, y = xy
    tw = dr.textlength(text, font=font)
    th = font.size
    extra = 18 if dot else 0
    dr.rounded_rectangle((x, y, x + tw + 2 * pad[0] + extra, y + th + 2 * pad[1]), radius=8, fill=fill)
    if dot:
        cy = y + pad[1] + th / 2 + 1
        dr.ellipse((x + pad[0], cy - 5, x + pad[0] + 10, cy + 5), fill=dot)
    dr.text((x + pad[0] + extra, y + pad[1] - 1), text, font=font, fill=ink)


def hud(img, c, grip, score, banner=None):
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    _chip(dr, (16, 16), TITLE[c], _font(24, True), dot=RGB[c])
    # squeeze meter
    x0, y0, bw, bh, fmax = W - 62, 84, 22, 220, 5.0
    dr.rounded_rectangle((x0 - 62, y0 - 40, x0 + bw + 10, y0 + bh + 40), radius=10, fill=(16, 18, 22, 190))
    dr.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=4, fill=(60, 62, 68, 255))
    f = min(grip / fmax, 1.0)
    if f > 0.01:
        col = (235, 104, 52) if grip < CRACK_N else (208, 59, 59)
        dr.rounded_rectangle((x0, y0 + bh * (1 - f), x0 + bw, y0 + bh), radius=4, fill=col)
    yc = y0 + bh * (1 - CRACK_N / fmax)
    dr.line((x0 - 6, yc, x0 + bw + 6, yc), fill=(250, 178, 25), width=3)
    sm = _font(14)
    dr.text((x0 - 54, yc - 9), "crack", font=sm, fill=(250, 178, 25))
    dr.text((x0 - 6, y0 - 30), f"{grip:.1f} N", font=sm, fill=(235, 235, 235))
    dr.text((x0 - 4, y0 + bh + 10), "grip", font=sm, fill=(200, 200, 200))
    # score
    f2 = _font(19, True)
    x = 16
    for o in ("success", "cracked", "dropped"):
        n = score.get(o, 0) + (score.get("timeout", 0) if o == "dropped" else 0)
        label = {"success": "in basket", "cracked": "cracked", "dropped": "dropped or stuck"}[o]
        t = f"{label} {n}"
        _chip(dr, (x, H - 52), t, f2, dot=OUT_RGB[o])
        x += dr.textlength(t, font=f2) + 54
    if banner:
        text, col = banner
        fb = _font(44, True)
        tw = dr.textlength(text, font=fb)
        _chip(dr, ((W - tw) / 2 - 18, H / 2 - 70), text, fb, ink=col, pad=(18, 12))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


# ---------------------------------------------------------------- carton runs
class _Cam:
    def __init__(self, env, w=W, h=H):
        self.r = mujoco.Renderer(env.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.15, -0.03, 0.07]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.60, 226, -36
        self.env = env

    def __call__(self):
        self.r.update_scene(self.env.d, camera=self.cam)
        return self.r.render().copy()


def run_carton(c, kinds, env_seed, render=True):
    env = EggEnv(seed=env_seed, render=render)
    env.reset(kinds=kinds)
    runner = Runner(_pol(c))
    cam = _Cam(env) if render else None
    frames, outs, score = [], [], {}
    st = dict(grip=0.0, banner=None, t=0)

    def grab():
        if render:
            b = st["banner"] if st["t"] > 0 else None
            st["t"] = max(0, st["t"] - 1)
            frames.append(hud(cam(), c, st["grip"], score, b))

    for _ in range(20):
        grab()
    for i, k in enumerate(PICK_ORDER):
        env.set_target(k, drop=DROP_SPOTS[i])

        def cb(obs, a, info):
            st["grip"] = info.grip_n
            grab()
        out, _ = policy_episode(env, runner, record=cb)
        outs.append(out)
        score[out] = score.get(out, 0) + 1
        st["banner"], st["t"], st["grip"] = (OUT_TXT[out], OUT_RGB[out]), 32, 0.0
        for _ in range(22):
            env.step(np.array([0, 0, 0.6, -1.0]))
            grab()
        for _ in range(36):
            env.step(np.r_[np.clip((HOME - env.ee()) / 0.0073, -1, 1), -1.0])
            grab()
        env.reset_arm()
    return frames, outs, score


def pick_carton(n=12):
    gain = json.loads((ROOT / "results/gain.json").read_text())
    avg = {c: next(r["success"] for r in gain if (r["cond"], r["n_robot"], r["seed"]) == SPEC[c]) for c in SPEC}
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(n):
        kinds = [str(x) for x in rng.permutation(["raw"] * 3 + ["stone"] * 3)]
        es = int(rng.integers(1000))
        res = {c: run_carton(c, kinds, es, render=False)[1] for c in SPEC}
        rows.append(dict(kinds=kinds, env_seed=es, **res))
        print("  carton", kinds, {c: res[c].count("success") for c in res}, flush=True)

    def dist(r):
        return sum(abs(r[c].count("success") - 6 * avg[c]) for c in SPEC) - 0.3 * min(r["A"].count("cracked"), 2)
    best = min(rows, key=dist)
    return best, avg, rows


def _card(lines, h=H, w=2 * W):
    im = Image.new("RGB", (w, h), (18, 20, 24))
    dr = ImageDraw.Draw(im)
    for text, xy, size, bold, col in lines:
        dr.text(xy, text, font=_font(size, bold), fill=col)
    return im


def title_card():
    return np.asarray(_card([
        ("EGG ROULETTE", (60, 170), 88, True, (255, 255, 255)),
        ("Six eggs that look the same. Some are raw (60 g). Some are painted stone (300 g).", (64, 300), 30, False,
         (225, 225, 225)),
        ("Squeeze too hard and a raw egg cracks. Too soft and the stone slips out.", (64, 345), 30, False,
         (225, 225, 225)),
        ("Both robots: the same SO-101 arm, trained on the same 25 teleoperation demos.", (64, 430), 22, False,
         (170, 170, 170)),
        ("The right one also learned from 150 human demos recorded with a touch-sensing glove.", (64, 462), 22,
         False, (170, 170, 170)),
        ("MuJoCo simulation", (64, 540), 18, False, (120, 120, 120)),
    ]))


def reveal_card(kinds, outs, avg):
    im = _card([("Same eggs, same robot. One was trained with human touch data.", (60, 60), 38, True,
                 (255, 255, 255))])
    dr = ImageDraw.Draw(im)
    f, fb = _font(22), _font(22, True)
    for i, kind in enumerate(kinds):
        x = 470 + i * 235
        dr.ellipse((x + 50, 150, x + 130, 255), fill=(246, 232, 205))
        dr.text((x + 90, 285), "RAW 60 g" if kind == "raw" else "STONE 300 g", font=fb, anchor="mm",
                fill=(255, 255, 255) if kind == "raw" else (160, 196, 255))
        for row, c in enumerate(("A", "C")):
            o = outs[c][i]
            dr.text((x + 90, 365 + row * 70), OUT_TXT[o].replace("couldn't lift it", "stuck"), font=fb,
                    anchor="mm", fill=OUT_RGB[o])
    for row, c in enumerate(("A", "C")):
        dr.ellipse((60, 356 + row * 70, 76, 372 + row * 70), fill=RGB[c])
        dr.text((88, 352 + row * 70), TITLE[c], font=f, fill=(225, 225, 225))
    dr.text((60, 500), f"Over 60 test eggs: robot demos only {100 * avg['A']:.0f}% intact in the basket, with human "
                       f"touch data {100 * avg['C']:.0f}%. This carton was picked to match those averages.",
            font=_font(19), fill=(210, 210, 210))
    dr.text((60, 545), "MuJoCo 3.17 simulation of an SO-101. Crack force scaled to the SO-101's small gripper.",
            font=_font(17), fill=(130, 130, 130))
    return np.asarray(im)


def video():
    MEDIA.mkdir(exist_ok=True)
    cached = ROOT / "results/video_carton.json"
    if cached.exists():                   # the carton search is deterministic: reuse it
        c = json.loads(cached.read_text())
        best, avg, rows = c["chosen"], c["averages"], c["all"]
    else:
        best, avg, rows = pick_carton()
    print("chosen", best["kinds"], best["env_seed"], {c: best[c] for c in SPEC}, flush=True)
    runs = {c: run_carton(c, best["kinds"], best["env_seed"]) for c in SPEC}
    left, right = runs["A"][0], runs["C"][0]
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))
    side = [np.concatenate([a, np.full((H, 4, 3), 10, np.uint8), b], 1)[:, 2:-2] for a, b in zip(left, right)]
    card = reveal_card(best["kinds"], {c: runs[c][1] for c in SPEC}, avg)
    title = title_card()
    path = MEDIA / "egg_roulette.mp4"
    wr = imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, macro_block_size=8,
                            pixelformat="yuv420p")
    for f in [title] * (FPS * 4) + side + [card] * (FPS * 6):
        wr.append_data(f)
    wr.close()
    imageio.imwrite(MEDIA / "side_by_side.png", side[int(len(side) * 0.62)])
    imageio.imwrite(MEDIA / "reveal.png", card)
    # README gif: the side-by-side only, sped up 2x, 720 px wide, 15 fps
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "4", "-i", str(path), "-t", f"{len(side) / FPS:.1f}",
                    "-vf", "setpts=0.5*PTS,fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128"
                    "[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(MEDIA / "egg_roulette.gif")], check=True)
    (ROOT / "results/video_carton.json").write_text(json.dumps(dict(chosen=best, averages=avg, all=rows), indent=1))
    print("wrote", path)


# ---------------------------------------------------------------- stills
HERO_CAMS = [([0.15, -0.03, 0.06], 0.55, 226, -33)]


def hero():
    """A 16:9 still: the touch policy carrying a stone egg over the basket, two eggs already in."""
    MEDIA.mkdir(exist_ok=True)
    env = EggEnv(seed=21, render=True)
    env.reset(kinds=["raw", "stone", "stone", "raw", "raw", "stone"])
    runner = Runner(_pol("C"))
    for i, k in enumerate(PICK_ORDER[:2]):
        env.set_target(k, drop=DROP_SPOTS[i])
        policy_episode(env, runner)
        for _ in range(20):
            env.step(np.array([0, 0, 0.6, -1.0]))
        env.reset_arm()
    env.set_target(PICK_ORDER[2], drop=DROP_SPOTS[2])
    runner.reset()
    obs = env.obs()
    r = mujoco.Renderer(env.m, 1080, 1920)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.19, -0.04, 0.045]
    cam.distance, cam.azimuth, cam.elevation = 0.40, 222, -30
    best = None
    for t in range(330):
        obs, info = env.step(runner.act(flat_obs(obs)))
        p = env.egg_pos()
        if p[2] > 0.06 and p[1] < -0.02 and best is None:
            best = t
        if best is not None and t >= best + 4:
            break
    for i, (look, dist, az, el) in enumerate(HERO_CAMS):
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, dist, az, el
        r.update_scene(env.d, camera=cam)
        imageio.imwrite(MEDIA / ("hero.png" if i == 0 else f"hero_alt{i}.png"), r.render())
    print("wrote", MEDIA / "hero.png")


def anatomy(seed=0, tries=40):
    """One raw and one stone test egg per policy, from the same setups, with per-step squeeze and egg height.
    Picks the first setup where the robot-only policy fails in its typical way and the touch policy succeeds."""
    out = {}
    for kind, a_fail in (("raw", "cracked"), ("stone", ("timeout", "dropped"))):
        rng = np.random.default_rng(77)
        for _ in range(tries):
            s = int(rng.integers(10_000))
            eps = {}
            for c in ("A", "C"):
                env = EggEnv(seed=s)
                setup_episode(env, np.random.default_rng(s), kind)
                rec = dict(grip=[], egg_z=[])

                def cb(obs, a, info):
                    rec["grip"].append(float(info.grip_n))
                    p = env.egg_pos() if not env.eggs[env.target].cracked else env.d.xpos[env.debris[env.target][0]]
                    rec["egg_z"].append(float(p[2]))
                o, _ = policy_episode(env, Runner(_pol(c)), record=cb)
                eps[c] = dict(rec, outcome=o)
            if eps["C"]["outcome"] == "success" and eps["A"]["outcome"] in a_fail:
                out[kind] = eps
                print(f"  anatomy {kind}: setup seed {s}, A {eps['A']['outcome']}, C success", flush=True)
                break
    (ROOT / "results/anatomy.json").write_text(json.dumps(out))


def main(names=None):
    for n in names or ["anatomy", "hero", "video"]:
        {"anatomy": anatomy, "hero": hero, "video": video}[n]()
