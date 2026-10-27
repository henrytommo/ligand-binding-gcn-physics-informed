# assembling a parsed ligand + pocket into a pyg Data object

import numpy as np
import torch
from torch_geometric.data import Data

from de import features
from de.config import (
    COINCIDENT_DISTANCE,
    DISTANCE_CUTOFF,
    INTERFACE_DISTANCE_CUTOFF,
    NODE_COFACTOR,
    NODE_LIGAND,
    NODE_METAL,
    NODE_PROTEIN,
    RESIDUE_OTHER,
)

# edge classes
EDGE_LIG_COVALENT = 0
EDGE_LIG_LIG = 1
EDGE_PROT_PROT = 2
EDGE_LIG_PROT = 3
EDGE_LIG_METAL = 4
EDGE_LIG_COFACTOR = 5
EDGE_HETERO = 6
N_EDGE_TYPES = EDGE_HETERO + 1

# what a binding energy may sum over, i.e. not ligand-ligand etc
INTERFACE_EDGES = (EDGE_LIG_PROT, EDGE_LIG_METAL, EDGE_LIG_COFACTOR)


def _pocket_duplicates(ligand_coords, pocket_coords):
    """
    Mask over POCKET atoms, true where one duplicates an earlier atom.
    """
    coords = torch.from_numpy(np.concatenate([ligand_coords, pocket_coords]))
    close = (torch.cdist(coords, coords) < COINCIDENT_DISTANCE).numpy()
    # column j is a duplicate only if some earlier i < j sits on it, so
    # np.triu keeps the first of each group
    duplicate = np.triu(close, k=1).any(axis=0)
    return duplicate[len(ligand_coords):]


def build_graph(ligand, pocket, label, node_feature_names=None, pdb_code=None):
    """
    One complex -> one Data, no edges.
    Ligand nodes come first and are never dropped, so bond indices carry over
    """
    if node_feature_names is None:
        node_feature_names = features.DEFAULT_NODE_FEATURES

    n_ligand = len(ligand["coords"])
    # both parsers derive every array from one source, so a disagreement is a
    # parser bug, not bad input. build.py catches it either way
    assert len(ligand["atomic_nums"]) == n_ligand
    assert (
        len(pocket["coords"])
        == len(pocket["atomic_nums"])
        == len(pocket["node_types"])
        == len(pocket["residue_types"])
    )

    keep = ~_pocket_duplicates(ligand["coords"], pocket["coords"])
    coords = np.concatenate(
        [ligand["coords"], pocket["coords"][keep]]
    ).astype(np.float32)
    atomic_nums = np.concatenate(
        [ligand["atomic_nums"], pocket["atomic_nums"][keep]]
    )
    node_type = np.concatenate(
        [np.full(n_ligand, NODE_LIGAND, dtype=np.int8), pocket["node_types"][keep]]
    )
    # the ligand belongs to no amino acid, so it takes the catch-all class
    residue_type = np.concatenate(
        [np.full(n_ligand, RESIDUE_OTHER, dtype=np.int8), pocket["residue_types"][keep]]
    )

    # only atomic_nums is read today; the rest is the surface features.py's
    # declared stubs will need
    ctx = {
        "atomic_nums": atomic_nums,
        "coords": coords,
        "node_type": node_type,
        "residue_type": residue_type,
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
        residue_type=torch.from_numpy(residue_type.astype(np.int64)),
        y=torch.tensor([label], dtype=torch.float32),
        # *_index so pyg offsets it correctly if a batch is built before add_edges
        lig_bond_index=torch.from_numpy(ligand["bonds"]),
        n_ligand=n_ligand,
        pdb_code=pdb_code,
    )


def _classify_edges(edge_index, node_type, lig_bond_index):
    """
    An edge class per edge, from the node classes at each end plus the ligand's
    real bonds. Later assignments win: HETERO goes down first and is overridden
    wherever a ligand is one end, covalent last of all.
    """
    src, dst = edge_index
    # gather the endpoint classes once, not once per comparison
    type_src, type_dst = node_type[src], node_type[dst]
    lig_src, lig_dst = type_src == NODE_LIGAND, type_dst == NODE_LIGAND
    prot_src, prot_dst = type_src == NODE_PROTEIN, type_dst == NODE_PROTEIN
    cof_src, cof_dst = type_src == NODE_COFACTOR, type_dst == NODE_COFACTOR
    met_src, met_dst = type_src == NODE_METAL, type_dst == NODE_METAL

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


def add_edges(
    data,
    distance_cutoff=DISTANCE_CUTOFF,
    interface_cutoff=INTERFACE_DISTANCE_CUTOFF,
    edge_feature_names=None,
):
    """
    Build the radius graph for one Data in place, setting edge_index, edge_attr
    and edge_type. Returns the same object, so it works as a pyg transform.

    Two radii: `interface_cutoff` for pairs with exactly one ligand end, and
    `distance_cutoff` for everything else. A pair's class follows from node_type
    alone, so which radius applies is known before any distance is compared -
    nothing circular about choosing the cutoff per class.

    Both directions of each pair are kept, as pyg message passing expects.
    """
    if edge_feature_names is None:
        edge_feature_names = features.DEFAULT_EDGE_FEATURES

    distance_matrix = torch.cdist(data.pos, data.pos)
    ligand = data.node_type == NODE_LIGAND
    # xor: EXACTLY one end a ligand atom, which is precisely INTERFACE_EDGES.
    # ligand-ligand keeps the narrow radius - it is intramolecular, and the
    # energy sums no term for it
    interface = ligand.unsqueeze(1) ^ ligand.unsqueeze(0)
    # a per-pair radius rather than two masked comparisons or-ed together, so
    # this stays correct if the interface radius is ever set BELOW the other
    within = distance_matrix < torch.where(interface, interface_cutoff, distance_cutoff)
    # clear the diagonal in place rather than filtering an edge list afterwards,
    # which would copy the whole list to drop the self-loops
    within.fill_diagonal_(False)
    edge_index = torch.stack(within.nonzero(as_tuple=True))

    distances = distance_matrix[edge_index[0], edge_index[1]]
    edge_type = _classify_edges(edge_index, data.node_type, data.lig_bond_index)

    edge_attr = features.build_features(
        features.EDGE_FEATURES, edge_feature_names, {"distances": distances.numpy()}
    )

    data.edge_index = edge_index
    data.edge_attr = torch.from_numpy(edge_attr)
    data.edge_type = edge_type
    return data
