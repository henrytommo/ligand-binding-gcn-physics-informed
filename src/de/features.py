# physics features for the graph nodes and edges
#
# to add one: write a function taking the context dict and returning an
# (N, width) array, then add it to NODE_FEATURES or EDGE_FEATURES. width is read
# off the array, so nothing else changes.
#
# node context: atomic_nums, coords, node_type, residue_type (combined, ligand
# first), the raw ligand/pocket dicts and n_ligand - the last two carry the rdkit
# mol and the biopandas frame. edge context: distances.

import numpy as np

from de.config import COINCIDENT_DISTANCE, N_ELEMENTS
from de.pocket import PERIODIC_TABLE


def _column(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


# ---------------------------------------------------------------- node features


def _atomic_num(ctx):
    """
    don;t use atomic num as a feature, meaningless. data.atomic_num carries
    the int64 for an nn.Embedding instead.
    """
    return _column(ctx["atomic_nums"])


# a table, not a call per atom
VDW_RADII = np.array(
    [PERIODIC_TABLE.GetRvdw(z) for z in range(N_ELEMENTS)], dtype=np.float32
)


def _vdw_radius(ctx):
    """van der Waals radius in angstrom, the sigma of a Lennard-Jones term"""
    return _column(VDW_RADII[ctx["atomic_nums"]])


def _partial_charge(ctx):
    """
    Not implemented. Gasteiger covers the ligand; the protein needs a residue
    template set (amber ff14SB or similar) picking first, on the same scale to be
    worth concatenating.
    """
    raise NotImplementedError("partial_charge: pick a protein charge set first")


def _hbond_donor_acceptor(ctx):
    """
    Not implemented. Two flags per atom. rdkit's Lipinski.HDonorSmarts covers the
    ligand; the protein needs a per-residue-atom table.
    """
    raise NotImplementedError("hbond_donor_acceptor: needs a donor/acceptor table")


NODE_FEATURES = {
    "atomic_num": _atomic_num,
    "vdw_radius": _vdw_radius,
    "partial_charge": _partial_charge,
    "hbond_donor_acceptor": _hbond_donor_acceptor,
}


DEFAULT_NODE_FEATURES = ("vdw_radius",)


# ---------------------------------------------------------------- edge features


def _clamped_r(ctx):
    """
    r, floored at COINCIDENT_DISTANCE - a bound rather than a divisor, since
    graph.py has already dropped anything closer. Caps inv_r at 2 and inv_r6 at
    64; a 1e-3 clamp would instead return 1e18, an amplifier rather than a guard.
    """
    return np.maximum(ctx["distances"], COINCIDENT_DISTANCE)


def _inv_r(ctx):
    """1/r, coulomb term"""
    return _column(1.0 / _clamped_r(ctx))


def _inv_r6(ctx):
    """1/r^6, attractive half of Lennard-Jones"""
    return _column(1.0 / _clamped_r(ctx) ** 6)


def _distance(ctx):
    """raw r, for an rbf expansion or to check the others against"""
    return _column(ctx["distances"])


# angstrom. equal to config.DISTANCE_CUTOFF
_REFERENCE_DISTANCE = 4.0


def _log_inv_r6(ctx):
    """
    -6 log(r / 4A), for the learned branch, so smaller range

    NOT for the physics branch so lennard-jones still makes sense
    """
    return _column(-6.0 * np.log(_clamped_r(ctx) / _REFERENCE_DISTANCE))


EDGE_FEATURES = {
    "inv_r": _inv_r,
    "inv_r6": _inv_r6,
    "log_inv_r6": _log_inv_r6,
    "distance": _distance,
}

# raw physics terms plus a conditioned one for the learned branch
DEFAULT_EDGE_FEATURES = ("inv_r", "inv_r6", "log_inv_r6")

# for the energy sum
PHYSICS_EDGE_FEATURES = ("inv_r", "inv_r6")


# ---------------------------------------------------------------------- builder


def build_features(registry, names, ctx):
    """
    Stack the named features into one (N, F) float32 array, in the order given.
    build.py records the names alongside the cache, so a cache built with one
    feature set is identifiable.
    """
    if not names:
        raise ValueError("no features requested")

    blocks = []
    for name in names:
        if name not in registry:
            raise KeyError(f"unknown feature {name!r}; have {sorted(registry)}")
        block = np.asarray(registry[name](ctx), dtype=np.float32)
        if block.ndim != 2:
            raise ValueError(
                f"feature {name!r} must return a 2D array, got shape {block.shape}"
            )
        blocks.append(block)

    return np.concatenate(blocks, axis=1)
