"""Run the experiment grid and save one JSON per run.

From the repo root:
    python -m experiments.runner                 # everything in the grid
    python -m experiments.runner --quick         # seed 0 only, for testing
    python -m experiments.runner --datasets ar1 --models lstm

Each run trains the model once on its (dataset, shift, severity, seed), then evaluates
Split CP and ACI at every gamma on those same predictions. Results go to
results/runs/{run_id}.json; runs whose JSON already exists are skipped, so a crashed
sweep resumes where it stopped (--force reruns them).
"""
import argparse
import functools
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch

from data.synthetic import generate_ar1
from data.uci import load_uci
from data.shifts import inject_shift
from data.splits import chronological_split, make_windows
from models.lstm import LSTMForecaster
from conformal.split import split_conformal
from conformal.aci import aci
from metrics.coverage import hits, rolling_coverage, recovery_time, longest_miss_streak, mean_width
from experiments.configs import expand_grid, n_experiments

RUNS_DIR = "results/runs"

# name -> factory(seed) returning an object with .fit(X, y) and .predict(X)
MODELS = {"lstm": lambda seed: LSTMForecaster(seed=seed)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@functools.lru_cache(maxsize=1)
def _uci_base():
    # load_uci_with_shift = load_uci + inject_shift; load_uci re-parses the raw file on
    # every call, so load it once per process and inject per run.
    return load_uci()


def load_series(cfg):
    """Return (series, shift_at) for a config."""
    if cfg["dataset"] == "ar1":
        return generate_ar1(cfg["n"], cfg["phi"], shift_type=cfg["shift_type"], speed=cfg["speed"],
                            severity=cfg["severity"], seed=cfg["seed"])
    if cfg["dataset"] == "uci":
        return inject_shift(_uci_base(), cfg["shift_type"], cfg["speed"], cfg["severity"],
                            seed=cfg["seed"])
    raise ValueError(f"unknown dataset {cfg['dataset']!r}")


def score(lo, hi, y, shift, window):
    """Metrics measured after the shift (same definitions as midpoint_sweep.py)."""
    h = hits(y, lo, hi)
    rt = recovery_time(rolling_coverage(h, window), shift)
    return dict(post_cov=float(h[shift:].mean()),
                recovery=None if rt is None else int(rt),   # None = never recovered
                streak=int(longest_miss_streak(h[shift:])),
                width=float(mean_width(lo[shift:], hi[shift:])))


def run_one(cfg):
    """Train once, then evaluate Split CP and ACI at every gamma. Returns the run record."""
    t0 = time.time()
    set_seed(cfg["seed"])
    series, shift_at = load_series(cfg)
    a, b = chronological_split(len(series))                  # chronological, never shuffled
    X, y = make_windows(series, cfg["L"])
    idx = np.arange(cfg["L"], len(series))                   # time index each row predicts
    tr, ca, te = idx < a, (idx >= a) & (idx < b), idx >= b
    model = MODELS[cfg["model"]](cfg["seed"]).fit(X[tr], y[tr])
    cal = np.abs(y[ca] - model.predict(X[ca]))
    preds, y_te, shift = model.predict(X[te]), y[te], shift_at - b

    metrics = [dict(method="Split CP", gamma=0.0,
                    **score(*split_conformal(preds, cal, cfg["alpha"]), y_te, shift, cfg["window"]))]
    for g in cfg["gammas"]:
        lo, hi, _ = aci(preds, y_te, cal, cfg["alpha"], g)
        metrics.append(dict(method="ACI", gamma=g, **score(lo, hi, y_te, shift, cfg["window"])))

    return dict(run_id=cfg["run_id"], config=cfg, shift_in_test=int(shift), n_test=int(len(y_te)),
                metrics=metrics, runtime_sec=round(time.time() - t0, 2),
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def is_done(path):
    """A run counts as done only if its JSON exists and parses (guards against a torn file)."""
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, ValueError):
        return False


def save(record, out_dir):
    path = os.path.join(out_dir, f"{record['run_id']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:                                # write-then-rename: no partial JSON
        json.dump(record, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="seed 0 only")
    ap.add_argument("--datasets", nargs="+", help="subset of datasets, e.g. ar1")
    ap.add_argument("--models", nargs="+", help="subset of models, e.g. lstm")
    ap.add_argument("--out", default=RUNS_DIR, help="output directory (default %(default)s)")
    ap.add_argument("--force", action="store_true", help="rerun runs that already have a JSON")
    args = ap.parse_args()

    grid = {}
    if args.datasets:
        grid["dataset"] = args.datasets
    if args.models:
        grid["model"] = args.models
    configs = expand_grid(quick=args.quick, grid=grid)
    os.makedirs(args.out, exist_ok=True)

    todo = [c for c in configs
            if args.force or not is_done(os.path.join(args.out, f"{c['run_id']}.json"))]
    print(f"{len(configs)} runs ({n_experiments(configs)} experiments); "
          f"{len(configs) - len(todo)} already done, {len(todo)} to run")

    t0 = time.time()
    for i, cfg in enumerate(todo, 1):
        rec = run_one(cfg)
        save(rec, args.out)
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {cfg['run_id']}  {rec['runtime_sec']:.0f}s  "
              f"elapsed {elapsed / 60:.1f}m, eta {eta / 60:.1f}m", flush=True)
    print("done")


if __name__ == "__main__":
    main()
