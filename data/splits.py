"""Chronological splits and sliding windows."""
import numpy as np


def chronological_split(n, train=0.6, cal=0.2):
    """Index boundaries for a 60/20/20 time-ordered split. Never shuffle."""
    a, b = int(train * n), int((train + cal) * n)
    return a, b


def make_windows(series, L):
    """X[i] = series[i : i+L], y[i] = series[i+L]. Target index of row i is i+L."""
    X = np.lib.stride_tricks.sliding_window_view(series[:-1], L)
    y = series[L:]
    return X.astype(np.float32), y.astype(np.float32)