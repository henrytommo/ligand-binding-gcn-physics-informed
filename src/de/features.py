# physics features for the graph nodes and edges
#
# to add one: write a function taking the context dict and returning an
# (N, width) array, then add it to NODE_FEATURES or EDGE_FEATURES. width is read
# off the array, so nothing else changes.
#
# node context: atomic_nums, coords, node_type (combined, ligand first), plus
# the raw ligand/pocket dicts and n_ligand, for a featurizer needing the rdkit
# mol or the biopandas frame. edge context: distances, edge_type.

import numpy as np

from pocket import PERIODIC_TABLE


# ---------------------------------------------------------------- node features


def _atomic_num(ctx):
    """atomic number, as a stand-in identity until there are real features"""
    return ctx["atomic_nums"].astype(np.float32).reshape(-1, 1)


def _vdw_radius(ctx):
    """van der Waals radius in angstrom, the sigma of a Lennard-Jones term"""
    radii = [PERIODIC_TABLE.GetRvdw(int(z)) for z in ctx["atomic_nums"]]
    return np.asarray(radii, dtype=np.float32).reshape(-1, 1)


def _partial_charge(ctx):
    """
    Not implemented. Gasteiger (Chem.AllChem.ComputeGasteigerCharges) covers the
    ligand; the protein needs a residue template set (amber ff14SB or similar)
    choosing first, on the same scale to be worth concatenating.
    """
    raise NotImplementedError("partial_charge: pick a protein charge set first")


def _hbond_donor_acceptor(ctx):
    """
    Not implemented. Two flags per atom. rdkit's Lipinski.HDonorSmarts covers
    the ligand; the protein needs a per-residue-atom table.
    """
    raise NotImplementedError("hbond_donor_acceptor: needs a donor/acceptor table")


NODE_FEATURES = {
    "atomic_num": _atomic_num,
    "vdw_radius": _vdw_radius,
    "partial_charge": _partial_charge,
    "hbond_donor_acceptor": _hbond_donor_acceptor,
}

DEFAULT_NODE_FEATURES = ("atomic_num", "vdw_radius")


# ---------------------------------------------------------------- edge features

# angstrom. no real pair is this close; it only guards the division
_MIN_DISTANCE = 1e-3


def _inv_r(ctx):
    """1/r, coulomb term"""
    r = np.maximum(ctx["distances"], _MIN_DISTANCE)
    return (1.0 / r).astype(np.float32).reshape(-1, 1)


def _inv_r6(ctx):
    """1/r^6, attractive half of Lennard-Jones"""
    r = np.maximum(ctx["distances"], _MIN_DISTANCE)
    return (1.0 / r**6).astype(np.float32).reshape(-1, 1)


def _distance(ctx):
    """raw r, for an rbf expansion or to sanity check the others against"""
    return ctx["distances"].astype(np.float32).reshape(-1, 1)


EDGE_FEATURES = {
    "inv_r": _inv_r,
    "inv_r6": _inv_r6,
    "distance": _distance,
}

# the two physics terms. raw "distance" is left out: every one of these is a
# function of r, so they are a basis for the network rather than extra
# information, and r itself adds no shape the other two do not already give
DEFAULT_EDGE_FEATURES = ("inv_r", "inv_r6")


# ---------------------------------------------------------------------- builder


def build_features(registry, names, ctx):
    """
    Stack the named features into one (N, F) float32 array, in the order given.

    The caller records the names and F alongside the cache, which is what keeps
    a cache built with one feature set from being reused for another.
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
