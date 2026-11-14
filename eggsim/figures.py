"""Every figure in figures/, from results/*.json. One visual system: same palette, type, and marks throughout.

Colour follows the training condition everywhere: A robot demos only (blue), B + human video (yellow),
C + human touch (orange), D robot touch sensor (aqua). Outcome colours are the status set and only mean
outcome: success, cracked, dropped, stuck.
"""
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .scene import ROOT  # noqa: E402

G = 9.81

RES, FIG = ROOT / "results", ROOT / "figures"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COND = {"A": "#2a78d6", "B": "#eda100", "C": "#eb6834", "D": "#1baf7a"}
LABEL = {"A": "Robot demos only", "B": "+ human video, no touch", "C": "+ human demos with touch",
         "D": "Touch sensor on robot, robot demos only"}
OUTCOME = {"success": "#0ca30c", "cracked": "#d03b3b", "dropped": "#ec835a", "timeout": "#fab219"}
OUTCOME_LABEL = {"success": "in the basket", "cracked": "cracked", "dropped": "dropped", "timeout": "stuck"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab",
                                                     "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200, "lines.solid_capstyle": "round",
})


def _load(name):
    return json.loads((RES / name).read_text())


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """95% interval for p1 - p2 (Newcombe's hybrid score method)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def _title(fig, title, subtitle=None, y=0.985):
    fig.text(0.012, y, title, fontsize=13, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.012, y - 0.055 * (5.2 / fig.get_figheight()), subtitle, fontsize=9.5, color=INK2, va="top")


def _note(fig, text):
    fig.text(0.012, 0.012, text, fontsize=7.5, color=MUTED, va="bottom")


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def _pooled(rows):
    k = sum(round(r["success"] * r["n"]) for r in rows)
    n = sum(r["n"] for r in rows)
    return k, n


# ---------------------------------------------------------------- 1. gain chart
def gain():
    R = _load("gain.json")
    grid = sorted({r["n_robot"] for r in R})
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    fig.subplots_adjust(left=0.075, right=0.665, top=0.82, bottom=0.14)
    ends = {}
    for c in "ADBC":
        m, lo, hi = [], [], []
        for n in grid:
            k, N = _pooled([r for r in R if r["cond"] == c and r["n_robot"] == n])
            a, b = wilson(k, N)
            m.append(100 * k / N); lo.append(100 * a); hi.append(100 * b)
        ax.fill_between(grid, lo, hi, color=COND[c], alpha=0.10, lw=0)
        ax.plot(grid, m, color=COND[c], lw=2.6 if c == "C" else 2, zorder=3, label=LABEL[c])
        ax.scatter(grid, m, s=42, color=COND[c], edgecolor=SURFACE, linewidth=1.6, zorder=4)
        ends[c] = m[-1]
    # direct end labels with leader lines, spaced so they never collide
    ys = dict(sorted(ends.items(), key=lambda kv: kv[1]))
    placed, last = {}, -99
    for c, y in ys.items():
        y2 = max(y, last + 7.5)
        placed[c], last = y2, y2
    for c, y in ends.items():
        ax.annotate(f"{LABEL[c]}  {y:.0f}%", xy=(grid[-1], y), xytext=(grid[-1] * 1.12, placed[c]),
                    fontsize=9, color=INK, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=COND[c], lw=1.2, shrinkA=0, shrinkB=4))
    ax.set_xscale("log")
    ax.set_xticks(grid, [str(n) for n in grid])
    ax.minorticks_off()
    ax.set_xlim(grid[0] * 0.85, grid[-1] * 1.08)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("robot teleoperation demos in the training set")
    ax.set_ylabel("eggs moved to the basket intact")
    n_seeds = len({r["seed"] for r in R})
    _title(fig, "10 robot demos plus human touch data beat 100 robot demos alone",
           f"Look-alike raw (60 g, cracks at 2.2 N) and stone (300 g) eggs. Mean of {n_seeds} seeds x 60 test eggs; "
           "band = 95% Wilson interval.")
    _note(fig, "MuJoCo 3.17, SO-101 (MuJoCo Menagerie). Human grip strategy fitted to 718 real glove grasps "
               "(OpenGraph Labs, CC-BY-4.0).")
    _save(fig, "gain_chart")


# ---------------------------------------------------------------- 2. failure modes
def failures(n_robot=25):
    R = [r for r in _load("gain.json") if r["n_robot"] == n_robot]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.70, bottom=0.25, wspace=0.1)
    conds = ["C", "D", "B", "A"]
    for ax, kind in zip(axes, ("raw", "stone")):
        for i, c in enumerate(conds):
            agg = {}
            for r in R:
                if r["cond"] == c:
                    for o, v in r["detail"][kind].items():
                        agg[o] = agg.get(o, 0) + v
            tot = sum(agg.values())
            x = 0
            for o in ("success", "cracked", "dropped", "timeout"):
                w = 100 * agg.get(o, 0) / tot
                if w <= 0:
                    continue
                ax.barh(i, w - 0.6, left=x + 0.3, height=0.56, color=OUTCOME[o], lw=0)
                if w >= 9:
                    ax.text(x + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=8.5,
                            color="white" if o in ("success", "cracked") else INK)
                x += w
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% of test eggs")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{kind} eggs" + (" (60 g, crack above 2.2 N)" if kind == "raw" else " (300 g, never crack)"),
                     fontsize=10, color=INK)
        ax.spines["left"].set_visible(False)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(conds)), [LABEL[c] for c in conds], fontsize=9, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME[o]) for o in OUTCOME]
    fig.legend(handles, [OUTCOME_LABEL[o] for o in OUTCOME], loc="lower center", ncol=4, bbox_to_anchor=(0.6, 0.0))
    _title(fig, "Without touch the robot learns one compromise squeeze",
           f"Every test egg at {n_robot} robot demos, 5 seeds pooled. Without touch in the training data, most raw "
           "eggs crack and 35 to 53% of stones fail.")
    _save(fig, "failure_modes")


# ---------------------------------------------------------------- 3. episode anatomy
def anatomy():
    A = _load("anatomy.json")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex="col", gridspec_kw=dict(height_ratios=[1.35, 1]))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.10, hspace=0.12, wspace=0.16)
    for j, kind in enumerate(("raw", "stone")):
        ax, bx = axes[0, j], axes[1, j]
        for c in ("A", "C"):
            ep = A[kind][c]
            t = np.arange(len(ep["grip"])) / 30
            ax.plot(t, ep["grip"], color=COND[c], lw=2, label=LABEL[c])
            bx.plot(t, np.array(ep["egg_z"]) * 1000, color=COND[c], lw=2)
            end = ep["outcome"]
            tx = t[-1]
            ax.scatter([tx], [ep["grip"][-1]], s=46, color=COND[c], edgecolor=SURFACE, lw=1.6, zorder=5)
            ax.annotate(OUTCOME_LABEL[end], (tx, ep["grip"][-1]), xytext=(-6, 8),
                        textcoords="offset points", fontsize=8.5, color=INK, ha="right")
        if kind == "raw":
            ax.axhline(2.2, color=OUTCOME["cracked"], lw=1.2)
            ax.text(0.02, 2.28, "shell cracks above 2.2 N", fontsize=8, color=INK2, transform=ax.get_yaxis_transform())
        else:
            mu = _load("physics.json")["robot_cone"]["carry"]["mu_eff"]
            need = 0.3 * G / (2 * mu)
            ax.axhline(need, color=MUTED, lw=1.2)
            ax.text(0.02, need + 0.1, f"needed to carry 300 g on this gripper: {need:.1f} N (measured)", fontsize=8,
                    color=INK2, transform=ax.get_yaxis_transform())
        ax.set_ylim(0, 6.2)
        ax.set_title(f"{kind} egg", color=INK, fontsize=10.5)
        bx.set_ylim(0, 110)
        bx.set_xlabel("time (s)")
    axes[0, 0].set_ylabel("squeeze at the pads (N)")
    axes[1, 0].set_ylabel("egg height (mm)")
    h = [plt.Line2D([], [], color=COND[c], lw=2) for c in ("A", "C")]
    fig.legend(h, [LABEL["A"], LABEL["C"]], loc="upper right", ncol=2, bbox_to_anchor=(0.97, 0.90))
    _title(fig, "Same arm, same eggs: the touch-trained policy starts gentle and tightens on load",
           "One test egg of each kind, 25 robot demos, seed 0. Robot-only squeezes about the same on both: "
           "enough to crack the raw egg, not enough to lift the stone.")
    _save(fig, "episode_anatomy")


# ---------------------------------------------------------------- 4. ablations
GROUPS = [
    ("Model", ["C memory 1 step", "C memory 6 steps", "C chunks of 16", "C no action chunks"]),
    ("Human data volume and quality", ["C 50 human demos", "C 300 human demos", "C best 150 of 300 demos",
                                       "C human demos on varied eggs", "C 30% heavy-handed demonstrators",
                                       "C 60% heavy-handed demonstrators", "C 60% heavy-handed, labelled"]),
    ("Labels", ["C + phase labels", "C + object labels", "C + slip labels", "C + all three labels",
                "C + failed demos, labelled", "C + failed demos, unlabelled"]),
    ("Glove timing and calibration", ["C touch lags video 100 ms", "C touch lags video 300 ms",
                                      "C glove at 10 Hz", "C glove at 5 Hz", "C glove off by up to 30%"]),
    ("Which touch channel", ["C load (shear) only", "C slip only", "C grip pressure only"]),
    ("Sensing on the robot at run time", ["C servo-current estimate", "C no touch sensor on robot",
                                          "C glove-trained, run on servo est."]),
]


XMIN = -40


def ablations():
    R = _load("ablations.json")
    by = {}
    for r in R:
        by.setdefault(r["name"], []).append(r)
    kb, nb = _pooled(by["C base"])
    rows = []
    for g, names in GROUPS:
        rows.append(("group", g))
        for n in names:
            if n in by:
                k, N = _pooled(by[n])
                lo, hi = newcombe(k, N, kb, nb)
                rows.append(("row", n, 100 * (k / N - kb / nb), 100 * lo, 100 * hi, 100 * k / N))
    fig_h = 0.27 * len(rows) + 1.8
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    fig.subplots_adjust(left=0.36, right=0.86, top=1 - 1.15 / fig_h, bottom=0.55 / fig_h)
    y = 0
    yt, yl = [], []
    for row in rows:
        if row[0] == "group":
            ax.text(-0.555, y, row[1], transform=ax.get_yaxis_transform(), fontsize=9.5, fontweight="bold",
                    color=INK, va="center")
        else:
            _, n, d, lo, hi, abs_ = row
            sig = hi < 0 or lo > 0
            col = COND["C"] if sig else MUTED
            if d < XMIN:                  # off the scale: an arrow at the edge, labelled with its value
                ax.scatter([XMIN + 1], [y], marker="<", s=60, color=col, zorder=3, clip_on=False)
                ax.text(XMIN + 3, y, f"{d:.0f} pts", fontsize=8.5, color=INK2, va="center")
            else:
                ax.plot([max(lo, XMIN), hi], [y, y], color=col, lw=2, solid_capstyle="round")
                ax.scatter([d], [y], s=44, color=col if sig else SURFACE, edgecolor=col, lw=1.6, zorder=3)
            ax.text(1.02, y, f"{abs_:.0f}%", transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK2,
                    va="center")
            yt.append(y)
            yl.append(n[2:])
        y += 1
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(yt, yl, fontsize=9)
    ax.set_ylim(y - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"change in eggs moved intact vs the touch baseline ({100 * kb / nb:.0f}%), percentage points")
    ax.text(1.02, -0.9, "absolute", transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED, va="center")
    ax.set_xlim(XMIN, 20)
    ax.spines["left"].set_visible(False)
    _title(fig, "What else limits the touch-trained policy", "25 robot + 150 human demos unless stated; 5 seeds x 60 "
           "eggs per row. Filled = 95% interval excludes zero (Newcombe).", y=0.995)
    _save(fig, "ablations")


# ---------------------------------------------------------------- 5. unseen eggs
def generalization():
    R = _load("ablations.json")
    names = [("A base", "Robot demos only"), ("C base", "+ human touch, 2 egg kinds"),
             ("C human demos on varied eggs", "+ human touch, varied eggs"),
             ("C servo-current estimate", "+ human touch, servo current at run time"),
             ("C no touch sensor on robot", "+ human touch, no sensor at run time")]
    suites = [("raw_fragile", "fragile raw\n60 g, cracks 1.6 N"), ("medium", "medium\n150 g, cracks 3.5 N"),
              ("stone_light", "light stone\n180 g"), ("stone_heavy", "heavy stone\n400 g")]
    M = np.zeros((len(names), len(suites)))
    for i, (n, _) in enumerate(names):
        for j, (s, _) in enumerate(suites):
            agg = {}
            for r in R:
                if r["name"] == n and r.get("gen"):
                    for o, v in r["gen"][s].items():
                        agg[o] = agg.get(o, 0) + v
            M[i, j] = 100 * agg.get("success", 0) / sum(agg.values()) if agg else np.nan
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    fig.subplots_adjust(left=0.36, right=0.98, top=0.72, bottom=0.17)
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.0f}%", ha="center", va="center",
                    fontsize=9.5, color="white" if M[i, j] > 55 else (MUTED if np.isnan(M[i, j]) else INK))
    ax.set_xticks(range(len(suites)), [s[1] for s in suites], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(names)), [n[1] for n in names], fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(suites)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="minor", length=0)
    _title(fig, "Eggs it never saw: varied demos help on new weights, at a cost on fragile eggs",
           "Success on 150 eggs per cell (5 seeds x 30). 'Varied eggs' = human demos on 50-360 g eggs with "
           "130-170 g held out.")
    _save(fig, "generalization")


# ---------------------------------------------------------------- 6. robustness
def robustness():
    R = _load("robustness.json")
    mus = sorted({r["mu"] for r in R})
    ms = sorted({r["mass_scale"] for r in R})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.70, bottom=0.17, wspace=0.08)
    for ax, c in zip(axes, ("A", "D", "C")):
        M = np.zeros((len(ms), len(mus)))
        for i, s in enumerate(ms):
            for j, mu in enumerate(mus):
                k, n = _pooled([r for r in R if r["cond"] == c and r["mu"] == mu and r["mass_scale"] == s])
                M[i, j] = 100 * k / max(n, 1)
        ax.imshow(M, cmap=BLUES, vmin=0, vmax=100, aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 55 else INK)
        ti, tj = ms.index(1.0), mus.index(0.8)
        ax.add_patch(plt.Rectangle((tj - 0.5, ti - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
        ax.set_xticks(range(len(mus)), [f"{m:g}" for m in mus], fontsize=8.5)
        ax.set_yticks(range(len(ms)), [f"x{s:g}" for s in ms], fontsize=8.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{LABEL[c]}\nmean over the grid {np.mean(M):.0f}%", fontsize=10, color=INK, linespacing=1.5)
        ax.set_xlabel("pad friction coefficient")
    axes[0].set_ylabel("egg mass (x nominal)")
    _title(fig, "The touch lead holds near training conditions and fades on slippery pads or new weights",
           "Policies from 25 robot demos, 5 seeds x 20 eggs per cell. Outlined = training conditions (mu 0.8, "
           "nominal mass). Crack force stays 2.2 N.")
    _save(fig, "robustness")


# ---------------------------------------------------------------- 7. physics validation
def physics():
    P = _load("physics.json")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.15, wspace=0.28)
    # (a) friction cone
    ax = axes[0]
    fc = P["friction_cone"]
    r = np.array(fc["ratios"])
    order = ["elliptic, impratio 10 (used)", "pyramidal, impratio 10", "elliptic, impratio 1",
             "pyramidal, impratio 1 (default)"]
    cols = [COND["C"], COND["D"], COND["A"], COND["B"]]
    for name, col in zip(order, cols):
        y = np.maximum(fc["curves"][name], 0.05)
        ax.plot(r, y, color=col, lw=2.4 if "used" in name else 1.8)
        ax.text(r[-1], y[-1] * 1.25, name, fontsize=7.8, color=INK, ha="right", va="bottom")
    ax.axvline(1.0, color=INK2, lw=1)
    ax.text(1.03, 0.07, "Coulomb limit\n2 mu F = m g", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 60)
    ax.set_yticks([0.1, 1, 10, 30], ["0.1", "1", "10", "dropped"])
    ax.set_xlabel("squeeze / static limit  (F / (m g / 2 mu))")
    ax.set_ylabel("slip in 0.5 s (mm)")
    ax.set_title("a  Friction cone on an isolated pad rig", fontsize=10, color=INK)
    # (b) timestep convergence
    ax = axes[1]
    cv = P["convergence"]
    dts = np.array(cv["dts"]) * 1000
    icol = {"euler": COND["A"], "implicitfast": COND["C"], "rk4": COND["D"]}
    for integ, col in icol.items():
        ax.plot(dts, cv["r_crit"][integ], color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2,
                label="euler (on top of implicitfast)" if integ == "euler" else integ,
                zorder=4 if integ == "euler" else 3, ls=(0, (3, 2)) if integ == "euler" else "-")
    ax.axhline(cv["theory"], color=INK2, lw=1)
    ax.text(dts[0], cv["theory"] + 0.006, f"rigid-body theory {cv['theory']:.3f}", fontsize=8, color=INK2)
    ax.axvline(2, color=MUTED, lw=1)
    ax.text(2.1, 1.255, "2 ms used", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(dts, [f"{d:g}" for d in dts])
    ax.minorticks_off()
    ax.set_ylim(1.14, 1.28)
    ax.set_xlabel("timestep (ms)")
    ax.set_ylabel("critical squeeze ratio during a fast lift")
    ax.set_title("b  Converged across timestep and integrator", fontsize=10, color=INK)
    ax.legend(loc="lower right", fontsize=8)
    # (c) the robot's own cone
    ax = axes[2]
    rc = P["robot_cone"]
    gx = np.linspace(0, 4.6, 10)
    ax.plot(gx, 2 * 0.8 * gx, color=INK2, lw=1)
    ax.text(2.05, 2 * 0.8 * 2.05 + 0.15, "Coulomb, pad mu 0.8", fontsize=8, color=INK2, ha="right")
    for mode, col, name in (("still", COND["D"], "lift, hold still"), ("carry", COND["C"], "lift and carry")):
        sq = np.array([p["squeeze"] for p in rc[mode]["points"]])
        wt = np.array([p["max_weight"] for p in rc[mode]["points"]])
        ax.plot(gx, 2 * rc[mode]["mu_eff"] * gx, color=col, lw=1.2, ls=(0, (3, 2)))
        ax.scatter(sq, wt, s=46, color=col, edgecolor=SURFACE, lw=1.6, zorder=4,
                   label=f"{name}: mu_eff {rc[mode]['mu_eff']:.2f}")
    for w, name in ((0.06 * G, "raw egg"), (0.3 * G, "stone egg")):
        ax.axhline(w, color=AXIS, lw=1)
        ax.text(4.5, w - 0.12, name, fontsize=8, color=INK2, ha="right", va="top")
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 8)
    ax.set_xlabel("squeeze held at the pads (N)")
    ax.set_ylabel("heaviest egg held (N)")
    ax.set_title("c  Full SO-101: heaviest egg per squeeze", fontsize=10, color=INK)
    ax.legend(loc="upper left", fontsize=8)
    t = P["touch_check"]
    sp = P["speed"]
    _title(fig, "Physics checks: the contact model holds and slips where theory says it should",
           f"Native touch sensors on the pad sites match mj_contactForce on all {t['n']:,} loaded substeps checked. "
           f"The env runs {sp['realtime_factor']:.0f}x real time on one core.")
    _save(fig, "physics_validation")


# ---------------------------------------------------------------- 8. human grip data
def human_grip():
    H = _load("human_grip.json")
    T = np.array(H["templates"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.08, wspace=0.18)
    ax = axes[0]
    x = np.arange(T.shape[1])
    ax.fill_between(x, H["template_p25"], H["template_p75"], color=COND["C"], alpha=0.12, lw=0)
    ax.plot(x, H["template_mean"], color=COND["C"], lw=2.2)
    for xv in (20, 80):
        ax.axvline(xv, color=AXIS, lw=1)
    for xv, name in ((10, "reach + contact"), (50, "lift + hold"), (90, "release")):
        ax.text(xv, 1.75, name, ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 1.9)
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("glove pressure / hold level")
    ax.set_title(f"Mean grasp of {len(T)} (band = interquartile)", fontsize=10, color=INK)
    ax = axes[1]
    stats = [("rise to hold", f"{H['rise_s_median']:.2f} s", "median"),
             ("overshoot", f"{100 * (H['overshoot_median'] - 1):.0f}%", "median"),
             ("grasps with catch-up squeezes", f"{100 * H['share_grasps_with_catchup']:.0f}%", "+25% in 0.2 s"),
             ("hold force spread", f"{H['hold_spread_p90_over_p10']:.1f}x", "p90 / p10 across grasps")]
    ax.axis("off")
    for i, (k, v, sub) in enumerate(stats):
        y = 0.92 - i * 0.26
        ax.text(0, y, v, fontsize=20, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
        ax.text(0.42, y - 0.01, k, fontsize=9.5, color=INK, transform=ax.transAxes, va="top")
        ax.text(0.42, y - 0.09, sub, fontsize=8.5, color=MUTED, transform=ax.transAxes, va="top")
    _title(fig, "How people grip: the strategy the simulated glove wearer copies",
           f"{H['n_events']} grasps segmented from {H['n_episodes']} episodes of 160-taxel glove data "
           "(OpenGraph Labs, Ego-Tactile Manipulation, CC-BY-4.0).")
    _save(fig, "human_grip")


ALL = dict(gain=gain, failures=failures, anatomy=anatomy, ablations=ablations, generalization=generalization,
           robustness=robustness, physics=physics, human_grip=human_grip)


def main(names=None):
    for n in names or ALL:
        try:
            ALL[n]()
        except FileNotFoundError as e:
            print(f"skip {n}: {e.filename} missing")
