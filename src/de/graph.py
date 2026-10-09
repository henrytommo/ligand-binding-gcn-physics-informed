# assembling a parsed ligand + pocket into a pyg Data object

import numpy as np
import torch
from torch_geometric.data import Data

import features
from config import (
    DISTANCE_CUTOFF,
    NODE_COFACTOR,
    NODE_LIGAND,
    NODE_METAL,
    NODE_PROTEIN,
)

# edge classes. the three a binding energy term acts on are the ligand's
# contacts outwards: LIG_PROT (~4% of a radius graph), LIG_METAL (coordination,
# with its own functional form) and LIG_COFACTOR. HETERO is everything else
# touching a metal or cofactor - protein to metal, metal to metal and so on -
# which is the pocket's environment rather than the binding itself
EDGE_LIG_COVALENT = 0
EDGE_LIG_LIG = 1
EDGE_PROT_PROT = 2
EDGE_LIG_PROT = 3
EDGE_LIG_METAL = 4
EDGE_LIG_COFACTOR = 5
EDGE_HETERO = 6


def build_graph(ligand, pocket, label, node_feature_names=None):
    """
    One complex -> one Data, with no edges.

    Edges are left out deliberately: a radius graph is ~14k directed edges per
    complex, ~3 GB over the set against ~170 MB for the point clouds alone.
    add_edges() builds them at load time, so the cutoff can change without a
    reparse.

    Ligand nodes come first, so the ligand's bond indices carry over unchanged.
    """
    if node_feature_names is None:
        node_feature_names = features.DEFAULT_NODE_FEATURES

    n_ligand = len(ligand["coords"])
    n_pocket = len(pocket["coords"])
    if len(ligand["atomic_nums"]) != n_ligand:
        raise ValueError(
            f"ligand arrays disagree: {n_ligand} coords, "
            f"{len(ligand['atomic_nums'])} elements"
        )
    if not n_pocket == len(pocket["atomic_nums"]) == len(pocket["node_types"]):
        raise ValueError(
            f"pocket arrays disagree: {n_pocket} coords, "
            f"{len(pocket['atomic_nums'])} elements, "
            f"{len(pocket['node_types'])} node types"
        )

    coords = np.concatenate([ligand["coords"], pocket["coords"]]).astype(np.float32)
    atomic_nums = np.concatenate([ligand["atomic_nums"], pocket["atomic_nums"]])
    node_type = np.concatenate(
        [np.full(n_ligand, NODE_LIGAND, dtype=np.int8), pocket["node_types"]]
    )

    ctx = {
        "atomic_nums": atomic_nums,
        "coords": coords,
        "node_type": node_type,
        "ligand": ligand,
        "pocket": pocket,
        "n_ligand": n_ligand,
    }
    x = features.build_features(features.NODE_FEATURES, node_feature_names, ctx)

    return Data(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(coords),
        atomic_num=torch.from_numpy(atomic_nums.astype(np.int64)),
        node_type=torch.from_numpy(node_type.astype(np.int64)),
        y=torch.tensor([label], dtype=torch.float32),
        # *_index so pyg offsets it correctly if a batch is built before add_edges
        lig_bond_index=torch.from_numpy(ligand["bonds"]),
        n_ligand=n_ligand,
    )


def _classify_edges(edge_index, node_type, lig_bond_index):
    """
    An edge class per edge, from the node classes at each end plus the ligand's
    real bonds. Assigned in order, so later ones win: HETERO goes down first and
    is overridden wherever a ligand is one end, with covalent last of all.
    """
    src, dst = edge_index
    lig_src, lig_dst = node_type[src] == NODE_LIGAND, node_type[dst] == NODE_LIGAND
    prot_src, prot_dst = node_type[src] == NODE_PROTEIN, node_type[dst] == NODE_PROTEIN
    cof_src, cof_dst = node_type[src] == NODE_COFACTOR, node_type[dst] == NODE_COFACTOR
    met_src, met_dst = node_type[src] == NODE_METAL, node_type[dst] == NODE_METAL

    edge_type = torch.full_like(src, EDGE_PROT_PROT)
    edge_type[cof_src | cof_dst | met_src | met_dst] = EDGE_HETERO
    edge_type[lig_src & lig_dst] = EDGE_LIG_LIG
    edge_type[(lig_src & prot_dst) | (prot_src & lig_dst)] = EDGE_LIG_PROT
    edge_type[(lig_src & cof_dst) | (cof_src & lig_dst)] = EDGE_LIG_COFACTOR
    # the ligand's own metal coordination: what the metal nodes were kept for,
    # and not the same physics as a protein-metal contact
    edge_type[(lig_src & met_dst) | (met_src & lig_dst)] = EDGE_LIG_METAL

    if lig_bond_index.numel():
        # flatten each pair to one integer, so both bond directions match in one isin
        n_nodes = node_type.numel()
        begin, end = lig_bond_index
        bond_keys = torch.cat([begin * n_nodes + end, end * n_nodes + begin])
        edge_type[torch.isin(src * n_nodes + dst, bond_keys)] = EDGE_LIG_COVALENT

    return edge_type


def add_edges(data, distance_cutoff=DISTANCE_CUTOFF, edge_feature_names=None):
    """
    Build the radius graph for one Data in place, setting edge_index, edge_attr
    and edge_type. Returns the same object, so it works as a pyg transform.

    Both directions of each pair are kept, as pyg message passing expects.
    """
    if edge_feature_names is None:
        edge_feature_names = features.DEFAULT_EDGE_FEATURES

    distance_matrix = torch.cdist(data.pos, data.pos)
    edge_index = (distance_matrix < distance_cutoff).nonzero(as_tuple=False).t()
    # mask out self-loops rather than subtract an identity matrix, which would
    # be a second N x N allocation
    edge_index = edge_index[:, edge_index[0] != edge_index[1]].contiguous()

    distances = distance_matrix[edge_index[0], edge_index[1]]
    edge_type = _classify_edges(edge_index, data.node_type, data.lig_bond_index)

    edge_attr = features.build_features(
        features.EDGE_FEATURES,
        edge_feature_names,
        {"distances": distances.numpy(), "edge_type": edge_type.numpy()},
    )

    data.edge_index = edge_index
    data.edge_attr = torch.from_numpy(edge_attr)
    data.edge_type = edge_type
    return data
