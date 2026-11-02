"""Report figures from the midpoint sweep.

Run from the repo root:
    python -m experiments.plots                 # all three figures
    python -m experiments.plots --skip-rolling  # CSV-only figures (no model training)

Reads results/midpoint_sweep.csv. The rolling-coverage figure needs the per-step
hit sequences, which the CSV does not store, so it re-runs the sweep's own
one_series() and caches the arrays in results/cache/rolling.npz.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.midpoint_sweep import GAMMAS, SHIFTS, SEEDS, ALPHA, WINDOW, one_series
from conformal.split import split_conformal
from conformal.aci import aci
from metrics.coverage import hits, rolling_coverage

CSV = "results/midpoint_sweep.csv"
FIG_DIR = "results/figures"
CACHE = "results/cache/rolling.npz"
ROLL_GAMMAS = [0.005, 0.05]

# One color per method / gamma, used by every figure.
SPLIT_COLOR = "#eb6834"                      # Split CP
GAMMA_COLORS = dict(zip(GAMMAS, ["#9ec5f0", "#6aa5e6", "#2a78d6", "#1c58a3", "#123a6e"]))
SHIFT_COLOR = "#c0272d"
INK, INK2, GRID = "#1a1a19", "#52514e", "#e3e2de"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 100, "savefig.dpi": 160,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

TITLES = {f"{st}-{sp}": f"{st.capitalize()} shift, {sp}" for st, sp in SHIFTS}


def load():
    df = pd.read_csv(CSV)
    df["unrecovered"] = df["recovery"].isna()
    return df


def _panels(figsize, **kw):
    fig, axes = plt.subplots(2, 2, figsize=figsize, **kw)
    return fig, dict(zip(TITLES, axes.ravel()))


def tradeoff(df):
    """Recovery time vs post-shift width, one point per gamma (mean +/- std over seeds)."""
    fig, axes = _panels((12, 9))
    placed = {}
    for name, ax in axes.items():
        d = df[df["shift"] == name]
        split_w = d[d.method == "Split CP"]["width"].mean()
        ax.axvline(split_w, color=SPLIT_COLOR, ls=":", lw=2)
        ax.text(split_w, 0.7, " Split CP width\n (never recovers)", color=SPLIT_COLOR, ha="left",
                va="top", fontsize=10, transform=ax.get_xaxis_transform())
        pts, notes = [], []                       # (gamma, x, y)
        for g in GAMMAS:
            r = d[(d.method == "ACI") & (d.gamma == g)]
            rec = r["recovery"].dropna()
            if rec.empty:
                continue
            pts.append((g, r["width"].mean(), rec.mean()))
            ax.errorbar(pts[-1][1], pts[-1][2], xerr=r["width"].std(ddof=0), yerr=rec.std(ddof=0),
                        fmt="none", ecolor=GAMMA_COLORS[g], elinewidth=1.2, capsize=3, alpha=0.6,
                        zorder=2)
            ax.scatter(*pts[-1][1:], s=90, color=GAMMA_COLORS[g], edgecolor="white", linewidth=1.5,
                       zorder=3)
            if r["unrecovered"].any():
                notes.append(f"γ={g}: {int(r['unrecovered'].sum())}/{len(r)} runs never recovered "
                             "(excluded from mean)")
        ax.margins(x=0.3, y=0.18)
        ax.set_ylim(bottom=0)
        if notes:
            ax.text(0.99, 0.97, "\n".join(notes), transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK2)
        placed[name] = pts
        ax.set_title(TITLES[name])
        ax.set_xlabel("post-shift mean width")
        ax.set_ylabel("recovery time (steps)")
    fig.suptitle("Faster recovery costs wider intervals (ACI, mean ± std over seeds)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # Label each point to its right; push labels apart vertically where points cluster.
    gap = 16                                      # minimum label spacing in pixels
    for name, ax in axes.items():
        pts = sorted(placed[name], key=lambda p: ax.transData.transform((p[1], p[2]))[1])
        ys = []
        for g, x, y in pts:
            py = ax.transData.transform((x, y))[1]
            ys.append(max(py, ys[-1] + gap) if ys else py)
        for (g, x, y), ly in zip(pts, ys):
            py = ax.transData.transform((x, y))[1]
            dy = (ly - py) * 72 / fig.dpi
            ax.annotate(f"γ={g}", (x, y), textcoords="offset points", xytext=(14, dy - 4),
                        fontsize=10.5, color=INK, zorder=5,
                        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8, shrinkA=0, shrinkB=6)
                        if abs(dy) > 6 else None)
    return fig


def coverage_bars(df):
    """Post-shift coverage: Split CP vs ACI at each gamma, dashed line at 1 - alpha."""
    fig, axes = _panels((12, 9), sharey=True)
    labels = ["Split CP"] + [f"γ={g}" for g in GAMMAS]
    colors = [SPLIT_COLOR] + [GAMMA_COLORS[g] for g in GAMMAS]
    for name, ax in axes.items():
        d = df[df["shift"] == name]
        groups = [d[d.method == "Split CP"]] + [d[(d.method == "ACI") & (d.gamma == g)] for g in GAMMAS]
        means = [r["post_cov"].mean() for r in groups]
        stds = [r["post_cov"].std(ddof=0) for r in groups]
        x = np.arange(len(groups))
        ax.bar(x, means, yerr=stds, color=colors, width=0.7, capsize=4,
               error_kw=dict(ecolor=INK2, elinewidth=1.2))
        for xi, m in zip(x, means):
            ax.text(xi, 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=10.5,
                    color="white" if xi == 0 or xi >= 3 else INK, fontweight="bold")
        ax.axhline(1 - ALPHA, color=INK, ls="--", lw=1.5)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(TITLES[name])
        ax.set_ylabel("post-shift coverage")
        ax.grid(axis="x", visible=False)
    axes["mean-abrupt"].text(len(labels) - 0.5, 1 - ALPHA + 0.015, f"target {1 - ALPHA:.2f}",
                             ha="right", va="bottom", fontsize=10.5)
    fig.suptitle("Post-shift coverage vs the 0.90 target (mean ± std over seeds)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def compute_rolling():
    """Rolling coverage for Split CP and ACI over every shift/seed; cached on disk."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["data"].item()
    data = {}
    for st, sp in SHIFTS:
        for seed in SEEDS:
            preds, y, cal, shift = one_series(st, sp, seed)
            rc = {"Split CP": rolling_coverage(hits(y, *split_conformal(preds, cal, ALPHA)), WINDOW)}
            for g in ROLL_GAMMAS:
                lo, hi, _ = aci(preds, y, cal, ALPHA, g)
                rc[f"ACI γ={g}"] = rolling_coverage(hits(y, lo, hi), WINDOW)
            data.setdefault(f"{st}-{sp}", dict(shift=shift, curves={}))
            for k, v in rc.items():
                data[f"{st}-{sp}"]["curves"].setdefault(k, []).append(v)
            print(f"rolling: {st}-{sp} seed {seed}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, data=np.array(data, dtype=object))
    return data


def rolling(data):
    """Rolling coverage over the test period; line = mean over seeds, band = ± std."""
    fig, axes = _panels((12, 9), sharex=True, sharey=True)
    colors = {"Split CP": SPLIT_COLOR,
              **{f"ACI γ={g}": GAMMA_COLORS[g] for g in ROLL_GAMMAS}}
    for name, ax in axes.items():
        shift = data[name]["shift"]
        for k, c in colors.items():
            arr = np.array(data[name]["curves"][k])
            m, s = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            t = np.arange(len(m))
            ax.fill_between(t, m - s, m + s, color=c, alpha=0.18, linewidth=0)
            ax.plot(t, m, color=c, lw=2, label=k)
        ax.axvline(shift, color=SHIFT_COLOR, lw=1.5)
        ax.axhline(1 - ALPHA, color=INK, ls="--", lw=1.2)
        ax.set_title(TITLES[name])
        ax.set_ylim(0.3, 1.02)
        ax.set_ylabel(f"rolling coverage (window {WINDOW})")
        ax.set_xlabel("test step")
    axes["mean-abrupt"].text(shift + 8, 0.33, "shift start", color=SHIFT_COLOR, fontsize=11, va="bottom")
    axes["mean-abrupt"].legend(loc="lower right", frameon=False)
    for ax in axes.values():
        ax.label_outer()
    fig.suptitle("Rolling coverage: ACI recovers after the shift, Split CP does not "
                 f"(mean ± std, {len(SEEDS)} seeds)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rolling", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load()
    figs = {"tradeoff_recovery_vs_width": tradeoff(df), "coverage_bars": coverage_bars(df)}
    if not args.skip_rolling:
        figs["rolling_coverage"] = rolling(compute_rolling())
    for name, fig in figs.items():
        path = f"{FIG_DIR}/{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("saved", path)


if __name__ == "__main__":
    main()
