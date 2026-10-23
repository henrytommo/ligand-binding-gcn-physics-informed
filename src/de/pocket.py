# parsing protein pocket .pdb files with biopandas

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem

from de.config import (
    N_ELEMENTS,
    NODE_COFACTOR,
    NODE_METAL,
    NODE_PROTEIN,
    RESIDUE_NAME_TO_INDEX,
    RESIDUE_OTHER,
)
from de.residues import chain_residue_keys

# only water removed. some other solvents sit in the pocket
SOLVENT_RESNAMES = frozenset({"HOH", "DOD"})

# H ad D removed
NON_HEAVY_ELEMENTS = frozenset({"H", "D"})

# metals by atomic number, metalloids excluded as they participate in covalent bonding
METAL_ATOMIC_NUMS = frozenset(
    {3, 4, 11, 12, 13}             # Li Be Na Mg Al
    | set(range(19, 32))           # K Ca, the first row transition metals, Ga
    | set(range(37, 51))           # Rb Sr, the second row, In Sn
    | set(range(55, 85))           # Cs Ba, the lanthanides, the third row, Tl-Po
    | set(range(87, 119))          # Fr Ra and the actinides
)

# classify as an array index
IS_METAL = np.zeros(N_ELEMENTS, dtype=bool)
IS_METAL[list(METAL_ATOMIC_NUMS)] = True

# uppercase keys - elements uppercase in pdb
PERIODIC_TABLE = Chem.GetPeriodicTable()
ELEMENT_TO_ATOMIC_NUM = {
    PERIODIC_TABLE.GetElementSymbol(z).upper(): z for z in range(1, N_ELEMENTS)
}


def _classify_atoms(atoms, atomic_nums):
    """
    NODE_PROTEIN for atoms of a chain residue (see de.residues), then NODE_METAL
    for metal atoms, then NODE_COFACTOR for the rest.
    Chain is last so if any fit into 2 categories, chain wins
    possible bug: free amino acid is classified as a protein
    """
    # zip as numpy for speed
    keys = list(
        zip(
            atoms["chain_id"].to_numpy(),
            atoms["residue_number"].to_numpy(),
            atoms["insertion"].to_numpy(),
        )
    )
    chain_keys = chain_residue_keys(keys, atoms["atom_name"].to_numpy())

    node_types = np.where(
        IS_METAL[atomic_nums], NODE_METAL, NODE_COFACTOR
    ).astype(np.int8)
    for i, key in enumerate(keys):
        if key in chain_keys:
            node_types[i] = NODE_PROTEIN
    return node_types


def parse_pocket(pdb_path):
    """
    Returns {"coords", "atomic_nums", "node_types", "residue_types", "atoms"}.

    Reads ATOM and HETATM: HETATM holds the catalytic metals, the cofactors,
    and modified chain residues, so ATOM by itslef loses most of the binding chem

    "residue_types" is amino acid identity per atom, so an oxygen in one amino
    acid behaves differently to one in another.  a residue outside AA3_TO_1 falls 
    to RESIDUE_OTHER rather than guess.

    Raises ValueError if nothing survives filtering, or on fake element
    """
    ppdb = PandasPdb().read_pdb(str(pdb_path))
    atoms = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]], ignore_index=True)
    # normlaise cols
    for column in ("element_symbol", "residue_name", "atom_name"):
        atoms[column] = atoms[column].str.strip().str.upper()

    atoms = atoms[
        ~atoms["residue_name"].isin(SOLVENT_RESNAMES)
        & ~atoms["element_symbol"].isin(NON_HEAVY_ELEMENTS)
    ]

    # alternate conformers: same atom but 2 locs, due to diffraction pattern
    # drop dupes
    atoms = atoms.drop_duplicates(
        subset=["chain_id", "residue_number", "insertion", "atom_name"],
        keep="first",
    ).reset_index(drop=True)

    if atoms.empty:
        raise ValueError("no heavy atoms left after filtering")

    unknown = sorted(set(atoms["element_symbol"]) - set(ELEMENT_TO_ATOMIC_NUM))
    if unknown:
        raise ValueError(f"unknown element symbol(s) {unknown}")

    atomic_nums = (
        atoms["element_symbol"].map(ELEMENT_TO_ATOMIC_NUM).to_numpy(dtype=np.int16)
    )
    residue_types = (
        atoms["residue_name"]
        .map(RESIDUE_NAME_TO_INDEX)
        .fillna(RESIDUE_OTHER)
        .to_numpy(dtype=np.int8)
    )
    return {
        "coords": atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32),
        "atomic_nums": atomic_nums,
        "node_types": _classify_atoms(atoms, atomic_nums),
        "residue_types": residue_types,
        "atoms": atoms,
    }
