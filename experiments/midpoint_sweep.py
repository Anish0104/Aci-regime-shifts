"""Midpoint sweep: Split CP vs ACI across shift types, speeds and gammas.

Run from the repo root:
    python -m experiments.midpoint_sweep
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.synthetic import generate_ar1
from data.splits import chronological_split, make_windows
from models.lstm import LSTMForecaster
from conformal.split import split_conformal
from conformal.aci import aci
from metrics.coverage import hits, rolling_coverage, recovery_time, longest_miss_streak, mean_width

GAMMAS = [0.001, 0.005, 0.01, 0.05, 0.1]
SHIFTS = [(t, s) for t in ("mean", "variance") for s in ("abrupt", "gradual")]
SEEDS = [0, 1, 2]
L, ALPHA, WINDOW = 24, 0.1, 100


def one_series(shift_type, speed, seed):
    """Make one shifted series, train the LSTM, return test preds + calibration scores."""
    series, shift_at = generate_ar1(5000, 0.7, shift_type=shift_type, speed=speed,
                                    severity=2.0, seed=seed)
    a, b = chronological_split(len(series))
    X, y = make_windows(series, L)
    idx = np.arange(L, len(series))
    tr, ca, te = idx < a, (idx >= a) & (idx < b), idx >= b
    model = LSTMForecaster(seed=seed).fit(X[tr], y[tr])
    cal = np.abs(y[ca] - model.predict(X[ca]))
    return model.predict(X[te]), y[te], cal, shift_at - b


def score(lo, hi, y, shift):
    """Metrics measured after the shift."""
    h = hits(y, lo, hi)
    rt = recovery_time(rolling_coverage(h, WINDOW), shift)
    return dict(post_cov=float(h[shift:].mean()),
                recovery=np.nan if rt is None else rt,
                streak=longest_miss_streak(h[shift:]),
                width=mean_width(lo[shift:], hi[shift:]))


def main():
    os.makedirs("results/figures", exist_ok=True)
    rows = []

    # Train the LSTM once per (shift, seed); gamma only changes the conformal step
    for st, sp in SHIFTS:
        for seed in SEEDS:
            preds, y, cal, shift = one_series(st, sp, seed)
            lo, hi = split_conformal(preds, cal, ALPHA)
            rows.append(dict(shift=f"{st}-{sp}", method="Split CP", gamma=0.0, seed=seed,
                             **score(lo, hi, y, shift)))
            for g in GAMMAS:
                lo, hi, _ = aci(preds, y, cal, ALPHA, g)
                rows.append(dict(shift=f"{st}-{sp}", method="ACI", gamma=g, seed=seed,
                                 **score(lo, hi, y, shift)))
            print(f"done {st}-{sp} seed {seed}")

    # Save every run
    with open("results/midpoint_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Summary table (averaged over seeds) + trade-off plot
    print(f"\n{'shift':18s}{'method':10s}{'gamma':>7s}{'post_cov':>10s}"
          f"{'recovery':>10s}{'unrec':>7s}{'width':>8s}")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (st, sp) in zip(axes, SHIFTS):
        name = f"{st}-{sp}"
        xs, ys = [], []
        for method, g in [("Split CP", 0.0)] + [("ACI", g) for g in GAMMAS]:
            r = [x for x in rows if x["shift"] == name and x["method"] == method and x["gamma"] == g]
            pc = np.mean([x["post_cov"] for x in r])
            rec = np.array([x["recovery"] for x in r], dtype=float)
            unrec = int(np.isnan(rec).sum())
            rt = np.nanmean(rec) if unrec < len(rec) else np.nan
            wd = np.mean([x["width"] for x in r])
            print(f"{name:18s}{method:10s}{g:7.3f}{pc:10.3f}{rt:10.1f}{unrec:7d}{wd:8.2f}")
            if method == "ACI":
                xs.append(wd)
                ys.append(rt)
                ax.annotate(f"γ={g}", (wd, rt), fontsize=7)
        ax.plot(xs, ys, "o-")
        ax.set_title(name)
        ax.set_xlabel("post-shift width")
    axes[0].set_ylabel("recovery time (steps)")
    fig.tight_layout()
    fig.savefig("results/figures/midpoint_tradeoff.png", dpi=130)
    print("\nSaved results/midpoint_sweep.csv and results/figures/midpoint_tradeoff.png")


if __name__ == "__main__":
    main()