# what a model has to beat, as specs rather than a special case in the driver
#
# the size baseline is the one that matters: pdbbind rewards "bigger ligand binds
# tighter", so a model learning only that can look respectable - an r2 near it
# means no chemistry was learned.

import numpy as np

from exp.spec import ModelSpec, y_of


def _sizes(graphs):
    """ligand heavy atom count per graph"""
    return np.array([float(g.n_ligand) for g in graphs])


def _build(params, dims, seed=0, edge_feature_names=None):
    """nothing to size: fit returns the constants, so the model is a dict"""
    return {}


def _fit_mean(model, train, val, params, target):
    return {"mean": float(y_of(train).mean())}


def _predict_mean(model, graphs, target):
    y = y_of(graphs)
    return y, np.full_like(y, model["mean"])


def _fit_ligand_size(model, train, val, params, target):
    # least squares fit of pKd on ligand heavy atom count, train only
    sizes = _sizes(train)
    design = np.stack([sizes, np.ones_like(sizes)], axis=1)
    slope, intercept = np.linalg.lstsq(design, y_of(train), rcond=None)[0]
    return {"slope": float(slope), "intercept": float(intercept)}


def _predict_ligand_size(model, graphs, target):
    return y_of(graphs), model["slope"] * _sizes(graphs) + model["intercept"]


BASELINE_MEAN = ModelSpec(
    name="baseline_mean", build=_build, fit=_fit_mean, predict=_predict_mean
)
BASELINE_LIGAND_SIZE = ModelSpec(
    name="baseline_ligand_size",
    build=_build,
    fit=_fit_ligand_size,
    predict=_predict_ligand_size,
)

BASELINES = {spec.name: spec for spec in (BASELINE_MEAN, BASELINE_LIGAND_SIZE)}
