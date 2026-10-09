# parsing protein pocket .pdb files with biopandas

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem

from config import NODE_COFACTOR, NODE_METAL, NODE_PROTEIN

# only water. crystallisation additives (GOL, EDO, SO4) stay as cofactor nodes,
# since some of them do sit in the pocket
SOLVENT_RESNAMES = frozenset({"HOH", "DOD"})

# H positions are inferred in x-ray structures anyway. D is neutron-structure
# deuterium, in no periodic table, so it must go before the lookup below
NON_HEAVY_ELEMENTS = frozenset({"H", "D"})

# all three mark a chain residue, whatever it is called - which is what tells a
# modified residue (MSE, KCX, SEP: HETATM but covalently in the backbone) from a
# cofactor. all three, because a lone calcium ion is also an atom named CA
BACKBONE_ATOMS = frozenset({"N", "CA", "C"})

# metals by atomic number, metalloids deliberately excluded - Se above all,
# since selenomethionine is a chain residue and not a coordinating ion. B, Si,
# Ge, As, Sb, Te are out for the same reason: they turn up in covalent groups
METAL_ATOMIC_NUMS = frozenset(
    {3, 4, 11, 12, 13}             # Li Be Na Mg Al
    | set(range(19, 32))           # K Ca, the first row transition metals, Ga
    | set(range(37, 51))           # Rb Sr, the second row, In Sn
    | set(range(55, 85))           # Cs Ba, the lanthanides, the third row, Tl-Po
    | set(range(87, 119))          # Fr Ra and the actinides
)

# uppercase keys, as the pdb element column is written, covering metals,
# halogens and Se. from rdkit rather than hand listed, and read through a dict so
# an unknown symbol raises our error, not rdkit's c++ post-condition violation
PERIODIC_TABLE = Chem.GetPeriodicTable()
ELEMENT_TO_ATOMIC_NUM = {
    PERIODIC_TABLE.GetElementSymbol(z).upper(): z for z in range(1, 119)
}


def _classify_atoms(atoms, atomic_nums):
    """
    NODE_PROTEIN for atoms of a residue with a complete N/CA/C backbone, then
    NODE_METAL for metal atoms, then NODE_COFACTOR for the rest.

    The chain test comes first and wins, so the Fe of a heme and a bare Zn2+ are
    both metals while the Se of selenomethionine stays protein. Structural
    rather than a residue-name whitelist, which goes stale against the pdb's
    thousand-plus modified residue codes.

    Caveat: a free amino acid bound as a cofactor passes the backbone test and
    is called protein. Rare, and hard to separate without the chain topology.
    """
    keys = list(zip(atoms["chain_id"], atoms["residue_number"], atoms["insertion"]))
    names = atoms["atom_name"].str.strip().str.upper()

    backbone_seen = {}
    for key, name in zip(keys, names):
        if name in BACKBONE_ATOMS:
            backbone_seen.setdefault(key, set()).add(name)

    node_types = []
    for key, atomic_num in zip(keys, atomic_nums):
        if BACKBONE_ATOMS <= backbone_seen.get(key, frozenset()):
            node_types.append(NODE_PROTEIN)
        elif atomic_num in METAL_ATOMIC_NUMS:
            node_types.append(NODE_METAL)
        else:
            node_types.append(NODE_COFACTOR)
    return np.array(node_types, dtype=np.int8)


def parse_pocket(pdb_path):
    """
    Returns {"coords", "atomic_nums", "node_types", "atoms"} for one pocket.

    Reads ATOM and HETATM: HETATM is where the catalytic metals (Zn, Mg, Mn, Fe,
    Ca), cofactors (PLP, NAG, LLP) and modified chain residues (MSE, KCX, SEP)
    live, so ATOM alone discards most of the chemistry that decides binding.

    "atoms" is the filtered frame, kept so a featurizer can reach residue names
    or b-factors without re-reading, and for a later sequence-identity split. Its
    element_symbol column is stripped and uppercased.

    Raises ValueError if nothing survives filtering, or on an element symbol
    outside the periodic table - which used to become atomic number 0, a node
    with real coordinates and no identity.
    """
    ppdb = PandasPdb().read_pdb(str(pdb_path))
    atoms = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]], ignore_index=True)
    # normalised once, so the rest of this function and anything reading the
    # returned frame can use the column as it stands
    atoms["element_symbol"] = atoms["element_symbol"].str.strip().str.upper()

    atoms = atoms[
        ~atoms["residue_name"].str.strip().str.upper().isin(SOLVENT_RESNAMES)
        & ~atoms["element_symbol"].isin(NON_HEAVY_ELEMENTS)
    ]

    # alternate conformers repeat an atom at two positions, duplicating the node
    # and leaving a spurious ~0.1 A edge between the copies. atom names are
    # unique within a residue, so first-occurrence keeps the conformer the file
    # lists first, and handles altloc codes beyond A/B
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
    return {
        "coords": atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32),
        "atomic_nums": atomic_nums,
        "node_types": _classify_atoms(atoms, atomic_nums),
        "atoms": atoms,
    }
