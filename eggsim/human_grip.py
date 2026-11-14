"""Extract human grip-force profiles from OpenGraph Labs Ego-Tactile Manipulation (CC-BY-4.0).

Each grasp event = a run of frames for one hand that starts at a `contact_on` anchor, passes
through `load` (the object is being lifted or worked), and ends at `release`. For each event we sum
the tactile channels of the grasping hand (raw ADC units, uncalibrated) and record:

  - rise time: contact to 90% of the hold level
  - overshoot: peak during the first 0.5 s of load / median hold level
  - hold level per object (relative across objects, same glove)
  - catch-up events: fast grip increases during hold (+25% within 0.2 s), the human response to slip

Output: results/human_grip.json (stats + resampled normalised templates), results/human_grip_events.csv.
Get the data with scripts/download_data.sh.
"""
import json

import numpy as np
import pandas as pd

from .scene import ROOT

OUT = ROOT / "results"
FPS = 30


def first(x):
    return x[0] if (hasattr(x, "__len__") and not isinstance(x, str)) else x


def load():
    df = pd.read_parquet(ROOT / "data/ogl/data.parquet")
    for c in ["annotation.anchor", "annotation.grasp_type", "annotation.hand", "annotation.target_object"]:
        df[c] = df[c].map(first)
    df["pL"] = np.stack(df["observation.tactile.left"].values).sum(1)
    df["pR"] = np.stack(df["observation.tactile.right"].values).sum(1)
    return df


def events(df, on=1500.0, min_s=1.0):
    """Per hand, a grasp = a contiguous run where that hand's summed pressure stays above `on`
    (with gaps under 0.15 s bridged). Labels: first 20% of frames before the plateau = contact,
    middle = load/hold, last frames after the plateau drops = release."""
    out = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index").reset_index(drop=True)
        for hand, col in (("right", "pR"), ("left", "pL")):
            p = g[col].values.astype(float)
            act = p > on
            # bridge short gaps
            gap = int(0.15 * FPS)
            idx = np.where(act)[0]
            if not len(idx):
                continue
            runs, s0, prev = [], idx[0], idx[0]
            for i in idx[1:]:
                if i - prev > gap:
                    runs.append((s0, prev)); s0 = i
                prev = i
            runs.append((s0, prev))
            for a, b in runs:
                if (b - a) / FPS < min_s:
                    continue
                a0 = max(0, a - int(0.3 * FPS)); b0 = min(len(p), b + int(0.3 * FPS))
                seg = p[a0:b0]
                lab = np.array(["load"] * len(seg), dtype=object)
                plateau = np.where(seg >= 0.5 * np.median(seg[(seg > on)]))[0]
                lab[: plateau[0]] = "contact_on"; lab[plateau[-1] + 1:] = "release"
                if (lab == "release").sum() < 2:
                    lab[-3:] = "release"
                objs = g["annotation.target_object"].iloc[a:b + 1]
                gts = g["annotation.grasp_type"].iloc[a:b + 1]
                out.append(dict(ep=int(ep), hand=hand, obj=objs.mode().iloc[0], grasp=gts.mode().iloc[0],
                                p=seg, labels=lab))
    return out


def resample(x, n=60):
    if len(x) < 2:
        return np.full(n, x[0] if len(x) else 0.0)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def analyse(evts):
    rows, templates = [], []
    for e in evts:
        p, lab = e["p"], e["labels"]
        load_idx = np.where(lab == "load")[0]
        hold = np.median(p[load_idx])
        if hold < 200:  # glove barely loaded: likely a touch, not a grasp
            continue
        c = np.where(lab == "contact_on")[0]
        r = np.where(lab == "release")[0]
        pre = p[: load_idx[0] + 1]
        reach = np.where(p >= 0.9 * hold)[0]
        rise = (reach[0] / FPS) if len(reach) else np.nan
        early = p[load_idx[0]: load_idx[0] + int(0.5 * FPS)]
        overshoot = early.max() / hold if len(early) else np.nan
        # catch-up: +25% within 0.2 s during load
        lp = p[load_idx]
        w = int(0.2 * FPS)
        catch = 0
        if len(lp) > w:
            # distinct events: grip jumps >25% of the hold level within 0.2 s, while already holding
            hit = ((lp[w:] - lp[:-w]) > 0.25 * hold) & (lp[:-w] > 0.6 * hold)
            catch = int(np.sum(hit[1:] & ~hit[:-1]) + (hit[0] if len(hit) else 0))
        rows.append(dict(ep=e["ep"], hand=e["hand"], obj=e["obj"], grasp=e["grasp"], hold=float(hold),
                         rise_s=float(rise), overshoot=float(overshoot), catchups=catch,
                         load_s=len(load_idx) / FPS, release_s=len(r) / FPS))
        templates.append(np.concatenate([resample(pre / hold, 20), resample(lp / hold, 60),
                                         resample(p[r] / hold, 20)]))
    return pd.DataFrame(rows), np.array(templates)


def main():
    df = load()
    evts = events(df)
    stats, T = analyse(evts)
    by_obj = (stats.groupby("obj")["hold"].agg(["median", "count"]).query("count >= 3")
              .sort_values("median", ascending=False))
    by_obj["rel"] = by_obj["median"] / by_obj["median"].median()
    summary = dict(
        source="OpenGraph Labs, Ego-Tactile Manipulation (CC-BY-4.0), huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation",
        n_frames=int(len(df)), n_episodes=int(df["episode_index"].nunique()), n_events=int(len(stats)),
        rise_s_median=float(stats["rise_s"].median()), rise_s_iqr=[float(x) for x in stats["rise_s"].quantile([.25, .75])],
        overshoot_median=float(stats["overshoot"].median()),
        overshoot_iqr=[float(x) for x in stats["overshoot"].quantile([.25, .75])],
        catchup_events_per_grasp=float(stats["catchups"].mean()),
        share_grasps_with_catchup=float((stats["catchups"] > 0).mean()),
        hold_spread_p90_over_p10=float(stats["hold"].quantile(.9) / stats["hold"].quantile(.1)),
        by_object={k: dict(median_hold=float(v["median"]), n=int(v["count"]), rel=float(v["rel"]))
                   for k, v in by_obj.iterrows()},
        template_mean=T.mean(0).tolist(), template_p25=np.percentile(T, 25, 0).tolist(),
        template_p75=np.percentile(T, 75, 0).tolist(),
        templates=T.tolist(),
    )
    (OUT / "human_grip.json").write_text(json.dumps(summary))
    stats.to_csv(OUT / "human_grip_events.csv", index=False)
    print({k: v for k, v in summary.items() if not k.startswith("template")})

