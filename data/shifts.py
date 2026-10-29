"""Inject controlled shifts without changing the training or calibration data."""
import numpy as np


def inject_shift(series, shift_type, speed, severity, shift_at=None, ramp=300, seed=0):
    """Return a shifted copy and its onset index.

    Scale is estimated from the first 60% only. Variance severity is a
    standard-deviation multiplier (>= 1); independent Gaussian noise supplies
    the additional variance. Gradual effects start at zero at shift_at and
    reach full strength at shift_at + ramp, matching generate_ar1.
    """
    x = np.array(series, dtype=float, copy=True)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("series must be a finite 1-D array with at least two values")
    if shift_type not in ("mean", "variance"):
        raise ValueError("shift_type must be 'mean' or 'variance'")
    if speed not in ("abrupt", "gradual"):
        raise ValueError("speed must be 'abrupt' or 'gradual'")
    if not np.isfinite(severity) or (shift_type == "variance" and severity < 1):
        raise ValueError("severity must be finite; variance severity must be >= 1")
    if not isinstance(ramp, (int, np.integer)) or ramp <= 0:
        raise ValueError("ramp must be a positive integer")
    n = len(x)
    test_start = int(0.8 * n)
    if shift_at is None:
        shift_at = test_start + (n - test_start) // 4
    if not isinstance(shift_at, (int, np.integer)):
        raise ValueError("shift_at must be an integer")
    assert test_start <= shift_at < n, "shift must start in the last 20%"
    scale = x[:int(0.6 * n)].std()
    steps = np.arange(n - shift_at)
    weight = np.ones(len(steps)) if speed == "abrupt" else np.clip(steps / ramp, 0, 1)
    if shift_type == "mean":
        x[shift_at:] += severity * scale * weight
    else:
        multiplier = 1 + weight * (severity - 1)
        noise_scale = scale * np.sqrt(multiplier**2 - 1)
        x[shift_at:] += np.random.default_rng(seed).standard_normal(len(steps)) * noise_scale
    return x, int(shift_at)
