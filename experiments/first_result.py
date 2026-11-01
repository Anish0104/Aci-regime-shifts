"""First result: Split Conformal breaks after a shift, ACI recovers.

Run from the repo root:
    python -m experiments.first_result
"""
import json
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

CFG = dict(n=5000, phi=0.7, shift_type="mean", speed="abrupt", severity=2.0,
           L=24, alpha=0.1, gammas=[0.0, 0.005, 0.05], window=100, seed=0)


def main(cfg=CFG):
    os.makedirs("results/figures", exist_ok=True)

    # 1. data
    series, shift_at = generate_ar1(cfg["n"], cfg["phi"], shift_type=cfg["shift_type"],
                                    speed=cfg["speed"], severity=cfg["severity"], seed=cfg["seed"])
    a, b = chronological_split(len(series))
    X, y = make_windows(series, cfg["L"])
    target_idx = np.arange(cfg["L"], len(series))           # time index each row predicts
    tr, ca, te = target_idx < a, (target_idx >= a) & (target_idx < b), target_idx >= b

    # 2. forecaster (trained on train only)
    model = LSTMForecaster(seed=cfg["seed"]).fit(X[tr], y[tr])
    cal_scores = np.abs(y[ca] - model.predict(X[ca]))       # nonconformity = |y - y_hat|
    preds, y_te = model.predict(X[te]), y[te]
    shift_in_test = shift_at - b

    # 3. conformal methods
    runs = {"Split CP": split_conformal(preds, cal_scores, cfg["alpha"])}
    for g in cfg["gammas"]:
        lo, hi, _ = aci(preds, y_te, cal_scores, cfg["alpha"], g)
        runs[f"ACI g={g}"] = (lo, hi)

    # sanity check: ACI with gamma=0 must equal Split CP exactly
    assert np.allclose(runs["Split CP"][0], runs["ACI g=0.0"][0]), "gamma=0 ACI != Split CP"

    # 4. metrics + plot
    results = {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t = np.arange(len(y_te))
    axes[0].plot(t, y_te, color="0.3", lw=0.6, label="true value")
    for name, (lo, hi) in runs.items():
        if name == "ACI g=0.0":
            continue
        h = hits(y_te, lo, hi)
        rc = rolling_coverage(h, cfg["window"])
        results[name] = dict(
            coverage=float(h.mean()),
            post_shift_coverage=float(h[shift_in_test:].mean()),
            recovery_time=recovery_time(rc, shift_in_test),
            longest_miss_streak=longest_miss_streak(h),
            mean_width=mean_width(lo, hi),
        )
        axes[1].plot(t, rc, lw=1.4, label=name)
        if name in ("Split CP", "ACI g=0.05"):
            axes[0].fill_between(t, lo, hi, alpha=0.25, label=f"{name} interval")
    for ax in axes:
        ax.axvline(shift_in_test, color="red", ls="--", lw=1)
    axes[1].axhline(1 - cfg["alpha"], color="k", ls=":", lw=1)
    axes[0].set_ylabel("value"); axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel(f"rolling coverage ({cfg['window']})"); axes[1].set_xlabel("test step")
    axes[1].set_ylim(0, 1.02); axes[1].legend(loc="lower left", fontsize=8)
    axes[0].set_title(f"AR(1), {cfg['speed']} {cfg['shift_type']} shift (red line): Split CP vs ACI")
    fig.tight_layout()
    fig.savefig("results/figures/first_result.png", dpi=130)

    with open("results/first_result.json", "w") as f:
        json.dump(dict(config=cfg, results=results), f, indent=2)
    for name, r in results.items():
        print(f"{name:12s} {r}")


if __name__ == "__main__":
    main()