# build the cached dataset: index labels + parsed structures -> pyg graphs

import time
from collections import Counter
from datetime import date

import torch

from de import features
from de.config import CACHE_FILE, INDEX_FILE_PL, PL_DIR, YEAR_RANGES
from de.graph import build_graph
from de.index import build_labels
from de.ligand import parse_ligand, quiet_rdkit_warnings
from de.pocket import parse_pocket


def iter_complex_dirs(wanted, year_ranges=YEAR_RANGES, pl_dir=PL_DIR):
    """
    Yield (pdb_code, complex_dir) for every wanted complex with a directory on
    disk. The directory name is the pdb code.
    """
    for year_range in year_ranges:
        year_dir = pl_dir / year_range
        if not year_dir.is_dir():
            continue
        for complex_dir in sorted(year_dir.iterdir()):
            if complex_dir.is_dir() and complex_dir.name in wanted:
                yield complex_dir.name, complex_dir


def iter_complexes(labels, year_ranges=YEAR_RANGES, pl_dir=PL_DIR):
    """Yield (pdb_code, sdf_path, pocket_path) for every labelled complex."""
    for pdb_code, complex_dir in iter_complex_dirs(labels, year_ranges, pl_dir):
        yield (
            pdb_code,
            complex_dir / f"{pdb_code}_ligand.sdf",
            complex_dir / f"{pdb_code}_pocket.pdb",
        )


def build_dataset(labels, year_ranges=YEAR_RANGES, node_feature_names=None):
    """
    Returns (graphs, failures).
    Each complex attempted independently, so won;t fail whole pipeline
    """
    if node_feature_names is None:
        node_feature_names = features.DEFAULT_NODE_FEATURES

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
                build_graph(
                    ligand,
                    pocket,
                    labels[pdb_code],
                    node_feature_names,
                    pdb_code=pdb_code,
                )
            )
        except Exception as exc:
            failures.append((pdb_code, stage, f"{type(exc).__name__}: {exc}"))

    return graphs, failures


def save_dataset(blob, path=CACHE_FILE):
    """
    Graphs, failures and the config they were built with, in one file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return path


def load_dataset(path=CACHE_FILE):
    """weights_only=False because the file holds Data objects, not tensors."""
    return torch.load(path, weights_only=False)


def main():
    """Parse, build, save. Returns the saved blob, so a notebook caller does not
    have to read 160 MB back off disk."""
    quiet_rdkit_warnings()
    started = time.perf_counter()
    kind = "Kd"
    node_feature_names = features.DEFAULT_NODE_FEATURES

    labels = build_labels(INDEX_FILE_PL, kind)
    graphs, failures = build_dataset(labels, node_feature_names=node_feature_names)
    elapsed = time.perf_counter() - started

    blob = {
        "graphs": graphs,
        "failures": failures,
        "config": {
            "kind": kind,
            "year_ranges": list(YEAR_RANGES),
            "index_file": str(INDEX_FILE_PL),
            "node_features": list(node_feature_names),
            "num_node_features": int(graphs[0].x.shape[1]) if graphs else 0,
            "n_graphs": len(graphs),
            "built": date.today().isoformat(),
        },
    }
    path = save_dataset(blob)

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

    print(f"\nprocessed in {elapsed:.1f}s, wrote {path}")
    return blob


if __name__ == "__main__":
    main()
