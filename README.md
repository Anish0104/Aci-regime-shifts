# ACI Regime Shifts

How quickly does Adaptive Conformal Inference (ACI) recover its target coverage after time-series regime shifts, and at what interval-width cost? We also test whether a learned-uncertainty score (CQR) or a pretrained forecaster (Chronos) reduces the need for online adaptation.

ML II, Fall 2026, Rutgers University

## Team and Ownership

| Member | Role | Owns |
|---|---|---|
| Anish Shirodkar | Project lead; modeling and methods | `models/`, `conformal/`, `metrics/`; repo and code reviews |
| Anand Patel | Data engineering | `data/` (AR(1) generator, UCI and ENTSO-E loaders), Chronos pipeline |
| Santanu Agarwal | Experiment infrastructure | `experiments/` (configs, runner, seeding), `results/`, plotting |

## Structure

- `data/` — synthetic AR(1) generator with controlled shifts; UCI Electricity and ENTSO-E loaders
- `models/` — LSTM, GRU, fixed-window MLP, CQR quantile head, Chronos wrapper
- `conformal/` — Split Conformal, ACI
- `metrics/` — coverage, rolling coverage, recovery time, interval width, miscoverage streaks
- `experiments/` — configs and experiment runner
- `results/` — one JSON per run (not committed)
- `notebooks/` — exploration and plots

## Workflow

1. Never push directly to `main`.
2. Create a branch per task: `git checkout -b <name>/<task>` (e.g. `anand/ar1-generator`).
3. Push and open a pull request; Anish reviews and merges.
4. Pull `main` before starting new work: `git checkout main && git pull`.

## Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Milestones

- Oct 20: Split Conformal visibly breaking on an injected shift (AR(1))
- Oct 27: ACI implemented and validated
- Nov 2: Midpoint check-in
- Dec 1: Final report and public repo
