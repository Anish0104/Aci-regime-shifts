"""Split Conformal Prediction."""
import numpy as np


def conformal_quantile(scores, alpha):
    """Finite-sample-corrected (1 - alpha) quantile of calibration scores."""
    n = len(scores)
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return 0.0
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def split_conformal(preds, cal_scores, alpha=0.1):
    """Fixed-width interval: pred +/- q for every test step."""
    q = conformal_quantile(cal_scores, alpha)
    return preds - q, preds + q