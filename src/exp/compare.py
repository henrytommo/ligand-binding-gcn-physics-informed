# experiments

import argparse
import time
from itertools import combinations

import pandas as pd

from de.build import load_dataset
from de.config import DISTANCE_CUTOFF, PROCESSED_DIR
from de.graph import add_edges
from de.split import holdout_cluster, load_split, split_dataset
from exp.baselines import BASELINES
from exp.gcn import MODELS
from exp.metrics import METRIC_NAMES
from exp.spec import evaluate

RESULTS_DIR = PROCESSED_DIR / "experiments"
REGISTRY = {**MODELS, **BASELINES}


def prepare(spec_edge_features, limit=None):
    """
    Cached graphs with edges built for one edge feature set
    Edges are not cached (quick to compute)
    """
    graphs = load_dataset()["graphs"]
    if limit:
        graphs = graphs[:limit]
    return [add_edges(g.clone(), edge_feature_names=spec_edge_features) for g in graphs]


def run(names, seeds, mode="split", limit=None, guard=None, max_epochs=None):
    specs = [REGISTRY[name] for name in names]
    clusters = load_split()["clusters"]

    # one prepared copy per distinct edge feature set
    prepared = {}

    records = []
    for seed in seeds:
        split_kwargs = {"seed": seed, "cluster_of": clusters}
        if guard is not None:
            split_kwargs["guard"] = guard

        for spec in specs:
            if spec.edge_features not in prepared:
                prepared[spec.edge_features] = prepare(spec.edge_features, limit)
            graphs = prepared[spec.edge_features]

            if mode == "holdout":
                train, val, test = holdout_cluster(
                    graphs, cluster_of=clusters, seed=seed
                )
            else:
                train, val, test = split_dataset(graphs, **split_kwargs)

            params = dict(spec.params)
            if max_epochs:
                params["max_epochs"] = max_epochs

            started = time.perf_counter()
            _, metrics = evaluate(spec, train, val, test, params=params, seed=seed)
            records.append(
                {
                    "seed": seed,
                    "model": spec.name,
                    "n_train": len(train),
                    "n_test": len(test),
                    # the cutoff is an experiment parameter now that edges are
                    # built at load time, so this layer is the one to record it
                    "cutoff": DISTANCE_CUTOFF,
                    "edge_features": "+".join(spec.edge_features),
                    "guard": guard,
                    "seconds": round(time.perf_counter() - started, 1),
                    **metrics,
                }
            )
            print(
                f"  seed {seed:>2}  {spec.name:<22}"
                + "  ".join(f"{k} {metrics[k]:.3f}" for k in METRIC_NAMES)
                + f"   [{records[-1]['seconds']:.0f}s]"
            )

    return pd.DataFrame(records)


def summarise(frame):
    """per-model mean +- std, then matched-pair differences on shared splits"""
    print(f"\n--- {frame['seed'].nunique()} seeds, mean +- std ---")
    for name in frame["model"].unique():
        rows = frame[frame["model"] == name]
        print(
            f"  {name:<22}"
            + "  ".join(
                f"{k} {rows[k].mean():.3f}+-{rows[k].std():.3f}" for k in METRIC_NAMES
            )
        )

    models = list(frame["model"].unique())
    if len(models) < 2 or frame["seed"].nunique() < 2:
        return
    print("\nmatched pairs (same split per seed):")
    wide = frame.pivot(index="seed", columns="model", values="rmse")
    for a, b in combinations(models, 2):
        diff = (wide[a] - wide[b]).dropna()
        print(
            f"  rmse {a} - {b}: {diff.mean():+.3f} +- {diff.std():.3f}, "
            f"{a} lower on {(diff < 0).sum()}/{len(diff)} seeds"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multi-seed model comparison")
    parser.add_argument(
        "models", nargs="*", default=list(REGISTRY), choices=list(REGISTRY)
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mode", choices=("split", "holdout"), default="split")
    parser.add_argument("--limit", type=int, help="use only the first N complexes")
    parser.add_argument("--guard", type=float, help="override the large-cluster guard")
    parser.add_argument("--max-epochs", type=int, dest="max_epochs")
    parser.add_argument("--tag", default="", help="suffix for the results filename")
    args = parser.parse_args()

    frame = run(
        args.models, range(args.seeds), args.mode, args.limit,
        args.guard, args.max_epochs,
    )
    summarise(frame)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"compare_{args.mode}{args.tag}.csv"
    frame.to_csv(path, index=False)
    print(f"\nwrote {path}")
