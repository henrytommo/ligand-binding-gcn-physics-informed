# regression metrics for pKd prediction

import numpy as np


METRIC_NAMES = ("rmse", "mae", "pearson", "spearman", "r2")


def _ranks(values):
    """
    average ranks, ties shared - what spearman needs.
    np.unique gives the group sizes in sorted order, so cumsum minus the size is
    each group's first rank and (size + 1) / 2 centres it on the group.
    """
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - counts + (counts + 1) / 2.0)[inverse]


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a**2).sum() * (b**2).sum())
    return float(a @ b / denominator) if denominator else 0.0


def regression_metrics(y_true, y_pred):
    """
    rmse and mae in pKd units; pearson and spearman for ranking quality; r2 as
    the fraction of variance explained.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    residual = y_pred - y_true
    variance = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "rmse": float(np.sqrt((residual**2).mean())),
        "mae": float(np.abs(residual).mean()),
        "pearson": _pearson(y_true, y_pred),
        "spearman": _pearson(_ranks(y_true), _ranks(y_pred)),
        "r2": float(1 - (residual**2).sum() / variance) if variance else 0.0,
    }


