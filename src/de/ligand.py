# parsing ligand .sdf files with rdkit

import numpy as np
from rdkit import Chem, RDLogger


def quiet_rdkit_warnings():
    """
    Silence rdApp.warning, keep rdApp.error: every pdbbind sdf is tagged 2D
    while carrying 3D coordinates, so every one of them warns.
    """
    RDLogger.logger().setLevel(RDLogger.ERROR)


def parse_ligand(sdf_path):
    """
    Returns {"coords", "atomic_nums", "bonds", "mol"} for one ligand, with
    bonds as (2, B) indexing into coords.

    Raises ValueError for anything rdkit cannot read, sanitise or place in 3D.
    """
    # rdkit raises OSError for a missing or empty file and yields None for one
    # it cannot sanitise; both are one failure to the caller
    try:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    except OSError as exc:
        raise ValueError(f"rdkit could not open the sdf: {exc}") from exc

    mol = next(iter(supplier), None)
    if mol is None:
        raise ValueError("rdkit could not read or sanitise the sdf")
    if mol.GetNumConformers() == 0:
        raise ValueError("sdf has no conformer")

    # removeHs leaves the few H rdkit will not drop (charged, isotope labelled);
    # the pocket is heavy atoms only, so match it. RemoveAllHs rebuilds bonds
    # and the conformer, so the indices below stay consistent
    mol = Chem.RemoveAllHs(mol)

    conformer = mol.GetConformer()
    if not conformer.Is3D():
        raise ValueError("sdf conformer is not 3D")

    coords = np.asarray(conformer.GetPositions(), dtype=np.float32)
    atomic_nums = np.array(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int16
    )
    # reshape so a bondless single-atom ligand still gives (2, 0)
    bonds = (
        np.array(
            [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
            dtype=np.int64,
        )
        .reshape(-1, 2)
        .T
    )
    return {
        "coords": coords,
        "atomic_nums": atomic_nums,
        "bonds": bonds,
        "mol": mol,
    }
