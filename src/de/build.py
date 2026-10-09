# build the cached dataset: index labels + parsed structures -> pyg graphs

import time
from collections import Counter
from datetime import date

import torch

import features
from config import CACHE_FILE, INDEX_FILE_PL, PL_DIR, YEAR_RANGES
from graph import build_graph
from index import build_labels
from ligand import parse_ligand, quiet_rdkit_warnings
from pocket import parse_pocket


def iter_complexes(labels, year_ranges=YEAR_RANGES, pl_dir=PL_DIR):
    """
    Yield (pdb_code, sdf_path, pocket_path) for every labelled complex with a
    directory on disk. The directory name is the pdb code.
    """
    for year_range in year_ranges:
        year_dir = pl_dir / year_range
        if not year_dir.is_dir():
            continue
        for complex_dir in sorted(year_dir.iterdir()):
            pdb_code = complex_dir.name
            if not complex_dir.is_dir() or pdb_code not in labels:
                continue
            yield (
                pdb_code,
                complex_dir / f"{pdb_code}_ligand.sdf",
                complex_dir / f"{pdb_code}_pocket.pdb",
            )


def build_dataset(
    kind="Kd",
    year_ranges=YEAR_RANGES,
    node_feature_names=None,
    index_file=INDEX_FILE_PL,
    labels=None,
):
    """
    Returns (graphs, failures).

    Each complex is attempted independently, so one unreadable file costs that
    complex alone. failures holds (pdb_code, stage, reason), so the drop-out can
    be counted by cause rather than guessed at.

    Pass `labels` to reuse a label table already built, instead of parsing the
    index a second time.
    """
    if node_feature_names is None:
        node_feature_names = features.DEFAULT_NODE_FEATURES
    if labels is None:
        labels = build_labels(index_file, kind)

    graphs, failures = [], []

    for pdb_code, sdf_path, pocket_path in iter_complexes(labels, year_ranges):
        stage = "paths"
        try:
            if not sdf_path.exists():
                raise FileNotFoundError(sdf_path.name)
            if not pocket_path.exists():
                raise FileNotFoundError(pocket_path.name)

            stage = "ligand"
            ligand = parse_ligand(sdf_path)

            stage = "pocket"
            pocket = parse_pocket(pocket_path)

            stage = "graph"
            graphs.append(
                build_graph(ligand, pocket, labels[pdb_code], node_feature_names)
            )
        except Exception as exc:
            failures.append((pdb_code, stage, f"{type(exc).__name__}: {exc}"))

    return graphs, failures


def save_dataset(graphs, failures, config, path=CACHE_FILE):
    """
    Graphs, failures and the config they were built with, in one file - no
    second format to keep in step, and a cache can be checked against the ask.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"graphs": graphs, "failures": failures, "config": config}, path
    )
    return path


def load_dataset(path=CACHE_FILE):
    """weights_only=False because the file holds Data objects, not tensors."""
    return torch.load(path, weights_only=False)


def main():
    quiet_rdkit_warnings()
    started = time.perf_counter()
    kind = "Kd"
    node_feature_names = features.DEFAULT_NODE_FEATURES

    labels = build_labels(INDEX_FILE_PL, kind)
    graphs, failures = build_dataset(
        kind=kind, node_feature_names=node_feature_names, labels=labels
    )
    elapsed = time.perf_counter() - started

    config = {
        "kind": kind,
        "year_ranges": list(YEAR_RANGES),
        "index_file": str(INDEX_FILE_PL),
        "node_features": list(node_feature_names),
        "num_node_features": int(graphs[0].x.shape[1]) if graphs else 0,
        "n_graphs": len(graphs),
        "built": date.today().isoformat(),
    }
    path = save_dataset(graphs, failures, config)

    n_attempted = len(graphs) + len(failures)
    print(f"{kind + ' labels in index:':<23}{len(labels)}")
    print(f"{'complexes attempted:':<23}{n_attempted}")
    print(f"{'graphs built:':<23}{len(graphs)}")
    print(f"{'failed:':<23}{len(failures)}")
    if failures:
        print("\nfailures by stage and reason:")
        reasons = Counter(
            (stage, reason.split(":")[0]) for _code, stage, reason in failures
        )
        for (stage, reason), count in reasons.most_common():
            print(f"  {count:>5}  {stage:<8} {reason}")

    if graphs:
        nodes = sum(graph.num_nodes for graph in graphs)
        pkd = torch.cat([graph.y for graph in graphs])
        print(f"\n{'nodes total:':<23}{nodes} ({nodes / len(graphs):.0f} per graph)")
        print(f"{'pKd range:':<23}{pkd.min():.2f} .. {pkd.max():.2f}")
        print(f"{'pKd mean:':<23}{pkd.mean():.2f}")

    print(f"\nwrote {path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
