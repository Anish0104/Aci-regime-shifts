"""Adaptive Conformal Inference, Gibbs & Candes (2021).

alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   err_t = 1 if y_t outside the interval.
Missing makes alpha_t smaller -> wider intervals; covering makes them narrower.
With gamma = 0 this is exactly Split Conformal (sanity check).
"""
import numpy as np
from conformal.split import conformal_quantile


def aci(preds, y, cal_scores, alpha=0.1, gamma=0.005):
    T = len(preds)
    lo, hi = np.empty(T), np.empty(T)
    alphas = np.empty(T)
    a = alpha
    for t in range(T):
        q = conformal_quantile(cal_scores, a)   # uses only past information
        lo[t], hi[t] = preds[t] - q, preds[t] + q
        alphas[t] = a
        err = float(not (lo[t] <= y[t] <= hi[t]))  # y_t is revealed only AFTER the interval
        a = a + gamma * (alpha - err)
    return lo, hi, alphas