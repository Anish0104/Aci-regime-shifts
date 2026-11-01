"""Coverage, width, recovery time and miscoverage streaks."""
import numpy as np


def hits(y, lo, hi):
    return ((y >= lo) & (y <= hi)).astype(float)


def rolling_coverage(h, window=100):
    """Trailing-window coverage; NaN until the window fills."""
    c = np.convolve(h, np.ones(window) / window, mode="full")[: len(h)]
    c[: window - 1] = np.nan
    return c


def recovery_time(roll_cov, shift_idx, threshold=0.88, hold=50):
    """Steps after the shift until rolling coverage first drops below threshold
    and then stays >= threshold for `hold` steps. None if it never recovers.
    Returns 0 if coverage never dropped below the threshold."""
    post = roll_cov[shift_idx:]
    below = np.where(post < threshold)[0]
    if len(below) == 0:
        return 0
    start = below[0]
    ok = post >= threshold
    run = 0
    for i in range(start, len(post)):
        run = run + 1 if ok[i] else 0
        if run >= hold:
            return i - hold + 1
    return None


def longest_miss_streak(h):
    best = cur = 0
    for v in h:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def mean_width(lo, hi):
    w = hi - lo
    return float(np.mean(w[np.isfinite(w)]))