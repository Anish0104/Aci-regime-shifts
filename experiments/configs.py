"""Experiment grid.

One config = one trained model (dataset, model, shift, speed, severity, seed). Gamma does
not change the model, so every config carries the full list of gammas and the runner
evaluates Split CP plus ACI at each of them on the same predictions.

    from experiments.configs import expand_grid
    configs = expand_grid()          # list of dicts, each with a unique "run_id"
"""
import itertools

GRID = dict(
    dataset=["ar1", "uci"],
    model=["lstm"],                       # GRU, MLP, CQR to be added
    shift_type=["mean", "variance"],
    speed=["abrupt", "gradual"],
    severity=[2.0, 4.0],
    gamma=[0.001, 0.005, 0.01, 0.05, 0.1],
    seed=[0, 1, 2, 3, 4],
)

# Fixed across the whole sweep (same as midpoint_sweep.py); stored in every run's JSON.
FIXED = dict(n=5000, phi=0.7, L=24, alpha=0.1, window=100)   # n, phi: ar1 only


def make_run_id(dataset, model, shift_type, speed, severity, seed):
    return f"{dataset}_{model}_{shift_type}_{speed}_s{float(severity)}_seed{seed}"


def expand_grid(quick=False, grid=None):
    """Return the list of run configs. quick=True keeps seed 0 only.

    Seed is the outermost loop, so a sweep that stops early still has every condition
    covered for the seeds it finished.
    """
    g = {**GRID, **(grid or {})}
    seeds = g["seed"][:1] if quick else g["seed"]
    configs = []
    for seed, dataset, model, shift_type, speed, severity in itertools.product(
            seeds, g["dataset"], g["model"], g["shift_type"], g["speed"], g["severity"]):
        configs.append(dict(
            run_id=make_run_id(dataset, model, shift_type, speed, severity, seed),
            dataset=dataset, model=model, shift_type=shift_type, speed=speed,
            severity=float(severity), seed=seed, gammas=list(g["gamma"]), **FIXED))
    assert len({c["run_id"] for c in configs}) == len(configs), "run_ids are not unique"
    return configs


def n_experiments(configs):
    """Experiments = (Split CP + one ACI per gamma) per run."""
    return sum(1 + len(c["gammas"]) for c in configs)


if __name__ == "__main__":
    cfgs = expand_grid()
    print(f"{len(cfgs)} runs, {n_experiments(cfgs)} experiments; first: {cfgs[0]['run_id']}")
