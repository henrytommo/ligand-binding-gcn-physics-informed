# parsing the pdbbind index files into pKd labels

import math
import re

from config import INDEX_FILE_PL

# a usable label is an exact measurement alone in the binding field, e.g.
# Kd=49uM. rejects free text (the index sometimes puts a Kd in the comment) and
# censored bounds (Kd<10uM)
BINDING_TOKEN_RE = re.compile(
    r"^(?P<kind>Kd|Ki|IC50)=(?P<val>\d+\.?\d*)(?P<unit>[munpf]M)$"
)

UNIT_TO_MOLAR = {
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}


def load_index(index_file=INDEX_FILE_PL):
    """
    Read an index .lst into (pdb_code, resolution, year, binding, comment) rows.

    Fields are positional with the comment last, so maxsplit=4 is the whole
    parse - the comment may contain anything, including its own "Kd=..." text.

    Raises on a non-blank line with too few fields rather than skipping it: a
    silently dropped row is a silently dropped label.
    """
    rows = []
    with open(index_file) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(maxsplit=4)
            if len(parts) < 4:
                raise ValueError(
                    f"{index_file}:{lineno} has {len(parts)} fields, expected 4 or 5"
                )
            if len(parts) == 4:
                parts.append("")
            rows.append(tuple(p.strip() for p in parts))
    return rows


def parse_binding_token(token, kind="Kd"):
    """
    'Kd=49uM' -> 4.31, i.e. pKd = -log10(Kd in molar), positive for sub-molar.
    None unless the token is an exact measurement of `kind`, so censored bounds
    and free text are both rejected.
    """
    match = BINDING_TOKEN_RE.match(token)
    if match is None or match.group("kind") != kind:
        return None
    molar = float(match.group("val")) * UNIT_TO_MOLAR[match.group("unit")]
    if molar <= 0:
        return None
    return -math.log10(molar)


def build_labels(index_file=INDEX_FILE_PL, kind="Kd"):
    """{pdb_code: pKd} for every complex with an exact `kind` measurement."""
    labels = {}
    for pdb_code, _resolution, _year, binding, _comment in load_index(index_file):
        pkd = parse_binding_token(binding, kind)
        if pkd is not None:
            labels[pdb_code] = pkd
    return labels
