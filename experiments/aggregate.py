"""Aggregate results/runs/*.json into results/summary.csv.

From the repo root:
    python -m experiments.aggregate

summary.csv has one row per (dataset, model, shift_type, speed, severity, method, gamma)
with mean and std over seeds (sample std, ddof=1; NaN with a single seed):
    post_cov_*   post-shift coverage
    recovery_*   recovery time in steps, over the seeds that recovered
    n_unrecovered  seeds whose rolling coverage never recovered (n_recovered = the rest)
    width_*      mean post-shift interval width
    streak_*     longest post-shift run of misses
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

RUNS_DIR = "results/runs"
OUT = "results/summary.csv"
KEYS = ["dataset", "model", "shift_type", "speed", "severity", "method", "gamma"]


def load_runs(runs_dir=RUNS_DIR):
    """One row per (run, method, gamma) across every JSON in runs_dir."""
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        cfg = rec["config"]
        for m in rec["metrics"]:
            rows.append(dict(
                run_id=rec["run_id"], dataset=cfg["dataset"], model=cfg["model"],
                shift_type=cfg["shift_type"], speed=cfg["speed"], severity=cfg["severity"],
                seed=cfg["seed"], method=m["method"], gamma=m["gamma"],
                post_cov=m["post_cov"],
                recovery=np.nan if m["recovery"] is None else m["recovery"],
                unrecovered=m["recovery"] is None, width=m["width"], streak=m["streak"]))
    if not rows:
        raise SystemExit(f"no run JSONs found in {runs_dir}")
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(KEYS, sort=True)
    out = pd.DataFrame({"n_seeds": g["seed"].nunique(),
                        "n_unrecovered": g["unrecovered"].sum().astype(int)})
    out["n_recovered"] = out["n_seeds"] - out["n_unrecovered"]
    for col in ["post_cov", "recovery", "width", "streak"]:
        out[f"{col}_mean"] = g[col].mean()               # NaN recoveries are skipped
        out[f"{col}_std"] = g[col].std()
    cols = ["n_seeds", "post_cov_mean", "post_cov_std", "recovery_mean", "recovery_std",
            "n_unrecovered", "n_recovered", "width_mean", "width_std", "streak_mean", "streak_std"]
    return out[cols].reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", default=RUNS_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    df = load_runs(args.runs)
    summary = summarize(df)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"{df['run_id'].nunique()} runs, {len(df)} experiments -> {len(summary)} rows in {args.out}")


if __name__ == "__main__":
    main()
