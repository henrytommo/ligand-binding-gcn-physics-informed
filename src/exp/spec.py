# similar to qsar-biodegradation, to help woth experiments

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np

from de import features
from exp.metrics import regression_metrics


def y_of(graphs):
    """the pKd labels of a list of Data, as a float64 array"""
    return np.array([float(g.y) for g in graphs])


class Target(NamedTuple):
    """
    pKd centring, fitted on train only.
    """

    mean: float
    std: float

    @classmethod
    def fit(cls, graphs):
        y = y_of(graphs)
        return cls(float(y.mean()), float(y.std()) or 1.0) # avoid zero division

    def encode(self, y):
        return (y - self.mean) / self.std

    def decode(self, z):
        return z * self.std + self.mean


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable  # (params, dims, seed, edge_feature_names) -> model
    fit: Callable  # (model, train, val, params, target) -> model
    predict: Callable  # (model, graphs, target) -> (y_true, y_pred) in pKd units
    params: dict = field(default_factory=dict)
    edge_features: tuple = features.DEFAULT_EDGE_FEATURES


def dims_of(graphs):
    """what a model needs to size its layers, read off a built graph"""
    graph = graphs[0]
    return {
        "node_feat_dim": int(graph.x.shape[1]),
        "edge_feat_dim": int(graph.edge_attr.shape[1]),
    }


class Result(NamedTuple):
    """
    One fit and score. The predictions come back too, so per-complex error can
    be analysed later without refitting; they are in the order of whichever set
    `evaluate`'s `score` named.
    """

    model: object
    metrics: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    # the fitted encoding, so a caller can predict on another set (train) with
    # the statistics this fit used, rather than refitting them and hoping
    target: Target


def evaluate(spec, train, val, test, params=None, seed=0, score="val"):
    """
    Fit on train, then score one set. Target statistics come from train alone,
    so nothing about val or test reaches the fit.

    score="val" is the development default and leaves the test holdout
    untouched; "test" is for the one final reading. val doubles as the
    early-stopping set, so a val metric is optimistic by however much 20-odd
    epochs of epoch selection buys - every variant pays the same bias, which is
    what makes val fair for comparing them and wrong for quoting absolutely.
    """
    params = {**spec.params, **(params or {})}
    target = Target.fit(train)
    model = spec.build(params, dims_of(train), seed, spec.edge_features)
    model = spec.fit(model, train, val, params, target)
    y_true, y_pred = spec.predict(model, {"val": val, "test": test}[score], target)
    return Result(model, regression_metrics(y_true, y_pred), y_true, y_pred, target)
