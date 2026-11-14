"""Procedural textures for the scene: oak table top and moulded-pulp carton. Run once; outputs are committed."""
import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets/textures"
rng = np.random.default_rng(7)


def smooth_noise(h, w, cell):
    """Value noise: a coarse random grid upsampled with bicubic interpolation."""
    g = rng.random((h // cell + 3, w // cell + 3)).astype(np.float32)
    img = Image.fromarray(g, mode="F").resize((w + 3 * cell, h + 3 * cell), Image.BICUBIC)
    return np.asarray(img)[cell:cell + h, cell:cell + w]


def fbm(h, w, cells=(256, 128, 64, 32, 16), gain=0.55):
    out, amp = np.zeros((h, w), np.float32), 1.0
    for c in cells:
        out += amp * (smooth_noise(h, w, c) - 0.5)
        amp *= gain
    return out


def wood(size=2048):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    warp = fbm(size, size) * 0.5
    # growth rings seen edge-on: bands across y, stretched along x, wobbled by low-frequency noise
    rings = np.sin(2 * np.pi * (y * 11 + warp * 1.2 + 0.15 * np.sin(x * 2.3)))
    rings = 0.5 + 0.5 * np.sign(rings) * np.abs(rings) ** 0.6
    pores = smooth_noise(size, size, 2)[:, :] * 0.5 + smooth_noise(size, size, 4) * 0.5
    streak = np.asarray(Image.fromarray(pores, mode="F").resize((size // 16, size), Image.BILINEAR)
                        .resize((size, size), Image.BICUBIC))
    t = 0.40 * rings + 0.45 * streak + 0.15 * (fbm(size, size, (64, 32)) + 0.5)
    light, dark = np.array([0.74, 0.58, 0.42]), np.array([0.62, 0.46, 0.32])
    return dark + (light - dark) * np.clip(t, 0, 1)[..., None]


def pulp(size=512):
    n = fbm(size, size, (32, 16, 8, 4, 2), gain=0.7)
    base = np.array([0.79, 0.72, 0.60])
    return base * (0.93 + 0.12 * n)[..., None]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, img in (("wood", wood()), ("pulp", pulp())):
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT / f"{name}.png", optimize=True)
        print("wrote", OUT / f"{name}.png")
