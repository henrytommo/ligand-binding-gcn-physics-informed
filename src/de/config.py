# paths, node classes and tunable constants for the preprocessing pipeline

from pathlib import Path

# from __file__, not the cwd, so paths work from a notebook as well as the CLI
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"

INDEX_FILE_PL = INPUT_DIR / "index" / "INDEX_general_PL.2020R1.lst"
PL_DIR = INPUT_DIR / "P-L"
YEAR_RANGES = ("1981-2000", "2001-2010", "2011-2019")

CACHE_FILE = PROCESSED_DIR / "pl_kd_pocket.pt"

# node classes, here rather than in pocket.py because pocket.py and graph.py
# both need them. metals are their own class because coordination is not the
# same functional form as a vdW contact - it is ~2 A rather than ~3.5, strongly
# directional, and much larger in energy - so an energy term wants to select it.
# NODE_COFACTOR is then the organic hetero groups (PLP, NAG, and the
# crystallisation additives), which behave like protein atoms chemically
NODE_LIGAND = 0
NODE_PROTEIN = 1
NODE_COFACTOR = 2
NODE_METAL = 3

# angstrom, non-covalent contact range. edges are built at load time, not
# cached, so changing this needs no reparse
DISTANCE_CUTOFF = 5.0

# note: the index records no assay temperature, so anything deriving dG from pKd
# is assuming 298.15 K. add the constant here when something needs it
