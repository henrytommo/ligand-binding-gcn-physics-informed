# experiments
# every run writes one directory under data/train, so nothing is overwritten

import argparse
import json
import platform
import time
from datetime import datetime
from itertools import combinations

import pandas as pd
import torch

from de.build import load_dataset
from de.config import DISTANCE_CUTOFF, INTERFACE_DISTANCE_CUTOFF, TRAIN_DIR
from de.graph import add_edges
from de.split import holdout_cluster, load_split, split_dataset
from exp.baselines import BASELINES
from exp.gcn import MODELS
from exp.metrics import METRIC_NAMES
from exp.spec import evaluate

REGISTRY = {**MODELS, **BASELINES}


def prepare(graphs, edge_feature_names, cutoff, interface_cutoff):
    """
    Cached graphs with edges built for one edge feature set. edges not cached
    """
    return [
        add_edges(
            g.clone(),
            distance_cutoff=cutoff,
            interface_cutoff=interface_cutoff,
            edge_feature_names=edge_feature_names,
        )
        for g in graphs
    ]


def _save(out_dir, frames, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


def _frames(records, curves, predictions, train_predictions):
    """one row per model-seed, per epoch, per scored complex, per train complex"""
    return {
        "metrics": pd.DataFrame(records),
        "curve": pd.DataFrame(curves),
        "predictions": pd.DataFrame(predictions),
        # its own file rather than extra rows in predictions.csv, which
        # everything downstream reads as "the scored set and nothing else"
        "train_predictions": pd.DataFrame(train_predictions),
    }


def _config(
    meta, specs, score, dataset_config, split_config, started, cutoff,
    interface_cutoff,
):
    """everything needed to trace a result back to the code and data"""
    return {
        **(meta or {}),
        "models": {
            spec.name: {
                "params": spec.params,
                "edge_features": list(spec.edge_features),
            }
            for spec in specs
        },
        "scored": score,
        "cutoff": cutoff,
        "interface_cutoff": interface_cutoff,
        "dataset": dataset_config,
        "split": split_config,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "written": datetime.now().isoformat(timespec="seconds"),
    }


def run(
    names, seeds, mode="split", limit=None, guard=None, max_epochs=None,
    patience=None, progress=False, score="val", out_dir=None, meta=None,
    cutoff=DISTANCE_CUTOFF, interface_cutoff=INTERFACE_DISTANCE_CUTOFF,
):
    """
    Fit every spec on every seed, scoring `score` ("val" while developing, so
    the test holdout stays unread). Returns {name: DataFrame} for metrics (one
    row per model-seed), curve (one row per epoch) and predictions (one row per
    scored complex), written to out_dir with config.json and the best weights.
    """
    specs = [REGISTRY[name] for name in names]
    split = load_split()
    clusters = split["clusters"]

    blob = load_dataset()
    raw = blob["graphs"][:limit] if limit else blob["graphs"]
    dataset_config = blob["config"]
    # one prepared copy per DISTINCT edge feature set - dict.fromkeys dedupes
    # while keeping the order, unprepared graphs go (save memory)
    prepared = {
        edge_features: prepare(raw, edge_features, cutoff, interface_cutoff)
        for edge_features in dict.fromkeys(spec.edge_features for spec in specs)
    }
    del raw, blob

    records, curves, predictions, train_predictions = [], [], [], []
    started_all = time.perf_counter()

    for seed in seeds:
        split_kwargs = {"seed": seed, "cluster_of": clusters}
        if guard is not None:
            split_kwargs["guard"] = guard

        for spec in specs:
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
            if patience:
                params["patience"] = patience
            params["progress"] = progress

            started = time.perf_counter()
            result = evaluate(
                spec, train, val, test, params=params, seed=seed, score=score
            )
            seconds = round(time.perf_counter() - started, 1)
            scored = {"val": val, "test": test}[score]
            # the baselines are dicts with no history, so every read is a getattr
            history = getattr(result.model, "history", None) or []
            best_epoch = getattr(result.model, "best_epoch", None)

            records.append(
                {
                    "seed": seed,
                    "model": spec.name,
                    # which set the metrics below are on, so a val number can
                    # never be mistaken for a holdout number later
                    "scored": score,
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    # the cutoffs are experiment parameters now that edges are
                    # built at load time, so record here
                    "cutoff": cutoff,
                    "interface_cutoff": interface_cutoff,
                    "edge_features": "+".join(spec.edge_features),
                    "guard": guard,
                    "epochs_run": len(history),
                    "best_epoch": best_epoch,
                    "best_val_loss": getattr(result.model, "best_val_loss", None),
                    "seconds": seconds,
                    **result.metrics,
                }
            )
            curves += [{"seed": seed, "model": spec.name, **row} for row in history]
            predictions += [
                {
                    "seed": seed, "model": spec.name, "scored": score,
                    "pdb_code": graph.pdb_code,
                    "y_true": float(true), "y_pred": float(predicted),
                }
                for graph, true, predicted in zip(scored, result.y_true, result.y_pred)
            ]
            # train as well, for recalibration
            train_true, train_pred = spec.predict(result.model, train, result.target)
            train_predictions += [
                {
                    "seed": seed, "model": spec.name, "scored": "train",
                    "pdb_code": graph.pdb_code,
                    "y_true": float(true), "y_pred": float(predicted),
                }
                for graph, true, predicted in zip(train, train_true, train_pred)
            ]
            if out_dir is not None and hasattr(result.model, "state_dict"):
                torch.save(
                    result.model.state_dict(), out_dir / f"{spec.name}_seed{seed}.pt"
                )

            stopped = "" if best_epoch is None else f" stop@{best_epoch}/{len(history)}"
            print(
                f"  seed {seed:>2}  {spec.name:<22}{score:<5}"
                + "  ".join(f"{k} {result.metrics[k]:.3f}" for k in METRIC_NAMES)
                + f"   [{seconds:.0f}s{stopped}]"
            )

            # written after every model-seed so don't lose a seed if it breaks later
            if out_dir is not None:
                _save(
                    out_dir,
                    _frames(records, curves, predictions, train_predictions),
                    _config(
                        meta, specs, score, dataset_config, split["config"],
                        started_all, cutoff, interface_cutoff,
                    ),
                )

    return _frames(records, curves, predictions, train_predictions)


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
    parser.add_argument(
        "--seed-list", type=int, nargs="+", dest="seed_list",
        help="explicit seeds instead of range(--seeds), to resume a killed run",
    )
    parser.add_argument("--mode", choices=("split", "holdout"), default="split")
    parser.add_argument("--limit", type=int, help="use only the first N complexes")
    parser.add_argument("--guard", type=float, help="override the large-cluster guard")
    parser.add_argument(
        "--cutoff", type=float, default=DISTANCE_CUTOFF,
        help="edge radius for pairs that are not ligand-to-other",
    )
    parser.add_argument(
        "--interface-cutoff", type=float, default=INTERFACE_DISTANCE_CUTOFF,
        dest="interface_cutoff",
        help="edge radius for ligand-to-other pairs, the set the energy sums",
    )
    parser.add_argument("--max-epochs", type=int, dest="max_epochs")
    parser.add_argument("--patience", type=int, help="override early stopping patience")
    parser.add_argument(
        "--progress", action="store_true", help="print a line per epoch"
    )
    parser.add_argument(
        "--score", choices=("val", "test"), default="val",
        help="which set to report; keep val until the experiments are finished",
    )
    parser.add_argument("--tag", default="", help="suffix for the run directory")
    args = parser.parse_args()

    if args.score == "test":
        print(
            "scoring the TEST holdout - every reading costs some of what makes "
            "it a holdout. use --score val unless this is the final run.\n"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (
        f"_{args.tag}" if args.tag else ""
    )
    out_dir = TRAIN_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}")

    seeds = args.seed_list if args.seed_list else range(args.seeds)
    frames = run(
        args.models, seeds, args.mode, args.limit, args.guard,
        args.max_epochs, args.patience, args.progress, args.score,
        out_dir=out_dir, meta={"run_id": run_id, "args": vars(args)},
        cutoff=args.cutoff, interface_cutoff=args.interface_cutoff,
    )
    summarise(frames["metrics"])
    print(f"\nwrote {out_dir}")
