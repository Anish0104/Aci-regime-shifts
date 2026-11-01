"""Synthetic AR(1) series with a controlled regime shift."""
import numpy as np


def generate_ar1(
    n=5000,
    phi=0.7,
    sigma=1.0,
    shift_type="mean",      # "mean" or "variance"
    speed="abrupt",         # "abrupt" or "gradual"
    severity=2.0,           # mean: jump in units of stationary std; variance: noise std multiplier
    shift_at=None,          # index where the shift starts (default: inside the test segment)
    ramp=300,               # length of a gradual shift
    seed=0,
):
    """Return (series, shift_index).

    x_t = mu_t + phi * (x_{t-1} - mu_{t-1}) + sigma_t * eps_t
    Mean shift changes mu_t; variance shift changes sigma_t.
    """
    rng = np.random.default_rng(seed)
    if shift_at is None:
        test_start = int(0.8 * n)
        shift_at = test_start + (n - test_start) // 4   # 25% into the test segment

    assert int(0.8 * n) <= shift_at < n, "shift must start in the last 20%"

    # weight goes 0 -> 1 at the shift (instantly or linearly over `ramp` steps)
    t = np.arange(n)
    if speed == "abrupt":
        w = (t >= shift_at).astype(float)
    elif speed == "gradual":
        w = np.clip((t - shift_at) / ramp, 0.0, 1.0)
    else:
        raise ValueError(speed)

    stationary_std = sigma / np.sqrt(1 - phi**2)
    mu = np.zeros(n)
    sig = np.full(n, sigma)
    if shift_type == "mean":
        mu = w * severity * stationary_std
    elif shift_type == "variance":
        sig = sigma * (1 + w * (severity - 1))
    else:
        raise ValueError(shift_type)

    x = np.zeros(n)
    x[0] = mu[0] + sig[0] * rng.standard_normal()
    for i in range(1, n):
        x[i] = mu[i] + phi * (x[i - 1] - mu[i - 1]) + sig[i] * rng.standard_normal()
    return x, shift_at
