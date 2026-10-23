# grouped train/val/test split, so near-duplicate proteins cannot straddle it

import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import torch

from de.build import iter_complex_dirs, load_dataset
from de.config import AA3_TO_1, PL_DIR, PROCESSED_DIR, YEAR_RANGES
from de.residues import chain_residue_keys

SPLIT_FILE = PROCESSED_DIR / "split.pt"

# 0.3 - 30% similar (wilight zone). cov_mode 1 is
# coverage of target, so a short domain inside a longer protein still matches;
# --cluster-mode 1 is connected component, i.e. conservative single linkage
MIN_SEQ_ID = 0.3
COVERAGE = 0.8
COV_MODE = 1
CLUSTER_MODE = 1

# SINGLE_STEP to avoid leakage we had earlier due to cascaded clsutering
SENSITIVITY = 7.5
MAX_SEQS = 2000
SINGLE_STEP = 1

# mmseqs defaults this to all system memory; the dataset needs a fraction of it
MEMORY_LIMIT = "4G"

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 0

# test holdout
TEST_SEED = 0

# a cluster bigger than this frac is excluded from test eligibility, 
# and in the development split it goes to train first
GUARD = 0.10


def extract_sequences(protein_pdb_path):
    """
    The distinct chain sequences of one structure, sorted. Duplicate chains
    collapse, so a homodimer and its monomer compare equal.
    One residue per CA
    The full _protein.pdb, not just pocket
    """
    keys, names, residues, seen = [], [], [], set()
    with open(protein_pdb_path, errors="replace") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            key = (line[21], line[22:26].strip(), line[26])
            keys.append(key)
            names.append(line[12:16].strip().upper())
            # alternate conformers repeat the CA. keyed on the residue, like pocket.py
            if names[-1] == "CA" and key not in seen:
                seen.add(key)
                residues.append((key, line[17:20].strip().upper()))

    chain_keys = chain_residue_keys(keys, names)
    chains = defaultdict(list)
    for key, resname in residues:
        if key in chain_keys or resname in AA3_TO_1:
            chains[key[0]].append(AA3_TO_1.get(resname, "X"))
    return tuple(sorted({"".join(c) for c in chains.values() if c}))


def build_sequences(pdb_codes, year_ranges=YEAR_RANGES, pl_dir=PL_DIR):
    """
    ({pdb_code: chain sequences}, failures) for every readable _protein.pdb. returns failures.
    """
    wanted = set(pdb_codes)
    sequences, failures = {}, []
    for pdb_code, complex_dir in iter_complex_dirs(wanted, year_ranges, pl_dir):
        try:
            seqs = extract_sequences(complex_dir / f"{pdb_code}_protein.pdb")
            if not seqs:
                raise ValueError("no CA atoms found")
            sequences[pdb_code] = seqs
        except Exception as exc:
            failures.append((pdb_code, f"{type(exc).__name__}: {exc}"))

    missing = sorted(wanted - set(sequences) - {c for c, _ in failures})
    failures += [(c, "FileNotFoundError: no complex directory") for c in missing]
    return sequences, failures


def _merge_groups(all_keys, groups):
    """
    {key: cluster_id}, every key sharing a group sharing an id. Ids renumbered from 0.
    """
    parent = {key: key for key in all_keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for group in groups:
        members = iter(group)
        first = next(members, None)
        if first is None:
            continue
        for other in members:
            root_a, root_b = find(first), find(other)
            if root_a != root_b:
                parent[root_b] = root_a

    renumber = {}
    return {
        key: renumber.setdefault(find(key), len(renumber)) for key in all_keys
    }


def mmseqs_version():
    """the installed mmseqs2 version, recorded so a split is reproducible"""
    out = subprocess.run(["mmseqs", "version"], capture_output=True, text=True)
    return out.stdout.strip()


def cluster_sequences(
    sequences,
    min_seq_id=MIN_SEQ_ID,
    coverage=COVERAGE,
    cov_mode=COV_MODE,
    cluster_mode=CLUSTER_MODE,
    sensitivity=SENSITIVITY,
    max_seqs=MAX_SEQS,
    single_step=SINGLE_STEP,
    memory_limit=MEMORY_LIMIT,
):
    """
    {pdb_code: cluster_id} from mmseqs2 easy-cluster.
    mmseqs clusters chains, but a split is assigned per complex, so complexes
    sharing a chain cluster are merged afterwards
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError("mmseqs not on PATH - brew install mmseqs2")

    owners = defaultdict(set)  # chain sequence -> the complexes holding it
    for pdb_code, chains in sequences.items():
        for chain in chains:
            owners[chain].add(pdb_code)
    unique_chains = list(owners)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        fasta = tmp / "chains.fasta"
        with open(fasta, "w") as f:
            # integer ids, so no pdb code has to survive a fasta header
            for i, chain in enumerate(unique_chains):
                f.write(f">{i}\n{chain}\n")

        subprocess.run(
            [
                "mmseqs", "easy-cluster",
                str(fasta), str(tmp / "clu"), str(tmp / "work"),
                "--min-seq-id", str(min_seq_id),
                "-c", str(coverage),
                "--cov-mode", str(cov_mode),
                "--cluster-mode", str(cluster_mode),
                "-s", str(sensitivity),
                "--max-seqs", str(max_seqs),
                "--single-step-clustering", str(single_step),
                "--split-memory-limit", memory_limit,
                "-v", "1",
            ],
            check=True, capture_output=True, text=True,
        )

        # clu_cluster.tsv is an adjacency list: representative, then member
        chain_cluster = {}
        with open(tmp / "clu_cluster.tsv") as f:
            for line in f:
                representative, member = line.split()
                chain_cluster[int(member)] = int(representative)

    merge = defaultdict(set)
    for i, chain in enumerate(unique_chains):
        merge[chain_cluster[i]].update(owners[chain])
    return _merge_groups(sequences, merge.values())


def _members_of(cluster_of):
    """{cluster_id: [pdb_code]}, in a deterministic order"""
    members = defaultdict(list)
    for pdb_code, cluster in cluster_of.items():
        members[cluster].append(pdb_code)
    return members


def _guard_threshold(cluster_of, fractions, guard):
    """a cluster bigger than this is too big to place by quota"""
    return guard * min(frac * len(cluster_of) for frac in fractions.values())


def test_clusters(cluster_of, fractions=FRACTIONS, guard=GUARD, test_seed=TEST_SEED):
    """
    The cluster ids reserved for test, from test_seed alone - so no development
    seed can move them. Clusters above the guard are left to the development
    pool: one large family would otherwise swamp a 10% holdout.
    """
    members = _members_of(cluster_of)
    quota = fractions["test"] * len(cluster_of)
    threshold = _guard_threshold(cluster_of, fractions, guard)

    # sorted, not dict order
    eligible = sorted(c for c, names in members.items() if len(names) <= threshold)
    random.Random(test_seed).shuffle(eligible)

    reserved, filled = set(), 0
    for cluster in eligible:
        if filled >= quota:
            break
        reserved.add(cluster)
        filled += len(members[cluster])
    return reserved


def assign_splits(
    cluster_of, fractions=FRACTIONS, seed=SEED, guard=GUARD, test_seed=TEST_SEED,
):
    """
    {pdb_code: split name}, every member of a cluster landing in the same split.

    Test is made first and depends on test_seed only, so every development
    seed sees the same holdout. The remaining clusters are shuffled by `seed`;
    those above `guard` x the smallest quota go to train first, and the rest go
    to whichever of train/val is furthest below quota in RELATIVE terms.
    """
    members = _members_of(cluster_of)
    reserved = test_clusters(cluster_of, fractions, guard, test_seed)
    threshold = _guard_threshold(cluster_of, fractions, guard)

    assignment = {}
    for cluster in reserved:
        assignment.update(dict.fromkeys(members[cluster], "test"))

    # train/val quotas renormalised over what test left behind
    development = {name: fractions[name] for name in ("train", "val")}
    remaining = len(cluster_of) - len(assignment)
    quota = {
        name: frac / sum(development.values()) * remaining
        for name, frac in development.items()
    }

    clusters = [members[c] for c in sorted(members) if c not in reserved]
    random.Random(seed).shuffle(clusters)

    filled = dict.fromkeys(development, 0)

    def place(cluster, name):
        assignment.update(dict.fromkeys(cluster, name))
        filled[name] += len(cluster)

    rest = []
    for cluster in clusters:
        if len(cluster) > threshold:
            place(cluster, "train")
        else:
            rest.append(cluster)
    for cluster in rest:
        place(cluster, min(development, key=lambda n: filled[n] / quota[n]))
    return assignment


def split_dataset(
    graphs, cluster_of=None, seed=SEED, guard=GUARD, fractions=FRACTIONS,
    path=SPLIT_FILE, test_seed=TEST_SEED,
):
    """
    (train, val, test) lists of Data. test is the same whatever `seed` is.
    """
    if cluster_of is None:
        cluster_of = load_split(path)["clusters"]
    assignment = assign_splits(cluster_of, fractions, seed, guard, test_seed)

    out = {name: [] for name in fractions}
    missing = [g.pdb_code for g in graphs if g.pdb_code not in assignment]
    if missing:
        raise ValueError(
            f"{len(missing)} graphs have no cluster (e.g. {missing[:3]}) - "
            f"rebuild {path.name} after rebuilding the dataset"
        )
    for graph in graphs:
        out[assignment[graph.pdb_code]].append(graph)
    return out["train"], out["val"], out["test"]


def holdout_cluster(
    graphs, cluster_id=None, cluster_of=None, seed=SEED, val_fraction=0.1,
    path=SPLIT_FILE,
):
    """
    (train, val, test) with one whole protein cluster as test, default the
    largest - to test on largest cluster
    """
    if cluster_of is None:
        cluster_of = load_split(path)["clusters"]
    if cluster_id is None:
        cluster_id = Counter(cluster_of.values()).most_common(1)[0][0]

    test = [g for g in graphs if cluster_of.get(g.pdb_code) == cluster_id]
    if not test:
        raise ValueError(f"cluster {cluster_id} holds no graphs")
    remaining = [g for g in graphs if cluster_of.get(g.pdb_code) != cluster_id]

    val_clusters, filled = set(), 0
    wanted = val_fraction * len(remaining)
    clusters = sorted({cluster_of[g.pdb_code] for g in remaining})
    random.Random(seed).shuffle(clusters)
    sizes = Counter(cluster_of[g.pdb_code] for g in remaining)
    for cluster in clusters:
        if filled >= wanted:
            break
        val_clusters.add(cluster)
        filled += sizes[cluster]

    train = [g for g in remaining if cluster_of[g.pdb_code] not in val_clusters]
    val = [g for g in remaining if cluster_of[g.pdb_code] in val_clusters]
    return train, val, test


def load_split(path=SPLIT_FILE):
    """{pdb_code: split name} plus the config it was built with"""
    return torch.load(path, weights_only=False)


def main(path=SPLIT_FILE):
    """Cluster, assign, save. Returns the saved blob, so a notebook caller does
    not have to read back what was just in memory."""
    pdb_codes = [graph.pdb_code for graph in load_dataset()["graphs"]]
    sequences, failures = build_sequences(pdb_codes)
    cluster_of = cluster_sequences(sequences)
    assignment = assign_splits(cluster_of)
    sizes = Counter(cluster_of.values())

    # the guarantee the whole module exists for
    splits_of_cluster = defaultdict(set)
    for pdb_code, name in assignment.items():
        splits_of_cluster[cluster_of[pdb_code]].add(name)
    straddling = [c for c, names in splits_of_cluster.items() if len(names) > 1]
    if straddling:
        raise ValueError(
            f"{len(straddling)} clusters span more than one split "
            f"(e.g. {straddling[:3]}) - the split would leak"
        )

    blob = {
        "split": assignment,
        "clusters": cluster_of,
        "failures": failures,
        # the tool version too: without it a split is re-runnable, not
        # reproducible
        "config": {
            "fractions": FRACTIONS, "seed": SEED, "test_seed": TEST_SEED,
            "guard": GUARD,
            "n_complexes": len(assignment), "n_clusters": len(sizes),
            "mmseqs_version": mmseqs_version(), "min_seq_id": MIN_SEQ_ID,
            "coverage": COVERAGE, "cov_mode": COV_MODE,
            "cluster_mode": CLUSTER_MODE, "sensitivity": SENSITIVITY,
            "max_seqs": MAX_SEQS, "single_step": SINGLE_STEP,
            "built": date.today().isoformat(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)

    print(
        f"{len(pdb_codes)} complexes, {len(failures)} without a sequence, "
        f"{len(set(sequences.values()))} unique, {len(sizes)} clusters "
        f"(largest {max(sizes.values())})"
    )
    per_split = Counter(assignment.values())
    for name in FRACTIONS:
        n = per_split[name]
        print(
            f"  {name:<6}{n:>5} ({n / len(assignment) * 100:4.1f}%, "
            f"target {FRACTIONS[name] * 100:.0f}%)"
        )
    print(f"no cluster spans a split. wrote {path}")
    return blob


if __name__ == "__main__":
    main()
