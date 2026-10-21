# paths, shared taxonomies and constants

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
# one timestamped subdirectory per training run: curves, metrics, predictions,
# the config that produced them and the best weights
TRAIN_DIR = DATA_DIR / "train"

INDEX_FILE_PL = INPUT_DIR / "index" / "INDEX_general_PL.2020R1.lst"
PL_DIR = INPUT_DIR / "P-L"
YEAR_RANGES = ("1981-2000", "2001-2010", "2011-2019")

CACHE_FILE = PROCESSED_DIR / "pl_kd_pocket.pt"

# NODE_COFACTOR is the organic hetero groups (PLP, NAG,
# crystallisation additives), which behave like protein atoms chemically
NODE_LIGAND = 0
NODE_PROTEIN = 1
NODE_COFACTOR = 2
NODE_METAL = 3
N_NODE_TYPES = NODE_METAL + 1

# atomic numbers 1..118 are an embedding not linear feature
N_ELEMENTS = 119

# 2 atoms closer than this (in angstrom) are same atoms - not 2 different
COINCIDENT_DISTANCE = 0.5

# angstrom, non-covalent bond range (vdw and h)
DISTANCE_CUTOFF = 4.0

# three letter residue names to one letter codes, shared by the residue 
# embedding and split.py's sequence extraction
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "KCX": "K", "CSO": "C",
    "CSS": "C", "CCS": "C", "CME": "C", "MLY": "K", "HYP": "P", "PCA": "Q",
    "SEC": "U", "PYL": "O",
    # modified residues mapped to their parents for mmseqs2 clustering
    # losese info, but still have the atom embeddings in the gcn
    "CAS": "C", "NLE": "L", "ABA": "A", "LLP": "K", "DAL": "A", "YCM": "C",
    "ALY": "K", "CSD": "C", "OCS": "C", "PHD": "D", "SMC": "C", "FTR": "W",
}

# one embedding row per one letter code, then RESIDUE_OTHER for ligand, metal, 
# cofactor, etc. ode_type is embedded alongside this so they stay distinguishable
RESIDUE_LETTERS = tuple(sorted(set(AA3_TO_1.values())))
RESIDUE_TO_INDEX = {letter: i for i, letter in enumerate(RESIDUE_LETTERS)}
RESIDUE_NAME_TO_INDEX = {
    name: RESIDUE_TO_INDEX[letter] for name, letter in AA3_TO_1.items()
}
RESIDUE_OTHER = len(RESIDUE_LETTERS)
N_RESIDUE_TYPES = RESIDUE_OTHER + 1
