# ligand-binding-gcn-physics-informed

Predicting ligand binding affinity (pKd) from PDBbind+ v2020.R1 protein-ligand
complexes, with a GCN whose readout adds a physics-style interaction energy
summed over interface contacts. `notes.md` for log.

## Layout

```
src/de/            dataset: index -> parsed structures -> pyg graphs -> split
  index.py         pdbbind index files -> pKd labels
  ligand.py        .sdf via rdkit        pocket.py  .pdb via biopandas
  graph.py         one complex -> one Data; add_edges builds the radius graph
  features.py      node and edge features (1/r, 1/r^6, ...)
  split.py         mmseqs2-clustered split, so homologues cannot straddle it
  config.py        paths, taxonomies, cutoffs
src/exp/           experiments
  gcn.py           the model and its specs (the ablation ladder)
  baselines.py     what a model has to beat
  compare.py       the driver: every spec x every seed
  spec.py          fit/predict interface      metrics.py  rmse, mae, r, rho, r2
data/input/        pdbbind index files and P-L structures (not in git)
data/processed/    pl_kd_pocket.pt (cached graphs), split.pt
data/train/        one timestamped directory per run, never overwritten
```

Each run directory holds `metrics.csv` (one row per model-seed), `curve.csv`
(per epoch), `predictions.csv` and `train_predictions.csv` (one row per
complex), `config.json` (code, data and parameter provenance) and
`<model>_seed<n>.pt` weights.

## Setup

activate a venv and install requirements.txt

`split.py` also needs `mmseqs2` on PATH. Everything run from `src/`.

## Pipeline

```bash
cd src
python -m de.build        # parse everything -> data/processed/pl_kd_pocket.pt
python -m de.split        # cluster and assign splits -> data/processed/split.pt
```

Both are cached; rerun only when the inputs or the features change.

## Running experiments

```bash
cd src
python -m exp.compare gcn_physics gcn_plain --seeds 5 --tag my_run
```

Models: `gcn_physics`, `gcn_plain`, `gcn_physics_blind`, `gcn_contact_sum`,
`gcn_plain_wide`, `baseline_mean`, `baseline_ligand_size`. Named specs only, so
a result always traces back to one.

Useful flags:

| flag | effect |
| --- | --- |
| `--seed-list 0 1 2` | explicit seeds instead of `range(--seeds)`; resumes a killed run |
| `--score val\|test` | **keep `val`.** the test holdout is read once, at the end |
| `--cutoff` / `--interface-cutoff` | edge radii, in angstrom; the second is for ligand-to-other pairs (both default 4.0) |
| `--limit N` / `--max-epochs N` | a fast smoke test |
| `--guard` / `--patience` / `--progress` | large-cluster guard, early stopping, per-epoch log |

Results are written after every model-seed, so a kill loses at most the
in-progress one. A full-dataset model-seed is roughly 7-25 minutes on CPU; MPS was
measured ~12x slower, so leave it on CPU for apple silicon
