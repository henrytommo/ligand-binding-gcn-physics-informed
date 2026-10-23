# what counts as a residue of the protein chain

from collections import defaultdict

# all three, so a modified residue (MSE, KCX, SEP: HETATM but covalently in the
# backbone) counts and a lone calcium ion does not, both being atoms named CA.
BACKBONE_ATOMS = frozenset({"N", "CA", "C"})


def chain_residue_keys(keys, atom_names):
    """
    The residue keys whose atoms include a complete N/CA/C backbone. `keys` is
    (chain, number, insertion) per atom, `atom_names` the stripped uppercase
    name; same length.
    """
    seen = defaultdict(set)
    for key, name in zip(keys, atom_names):
        if name in BACKBONE_ATOMS:
            seen[key].add(name)
    return {key for key, names in seen.items() if BACKBONE_ATOMS <= names}
