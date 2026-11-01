# MAMTL experiments

This repository tests whether material-class supervision improves binary surface-defect recognition, including on materials that are absent from training.

## Data

Keep DefectAtlas-20 outside Git with this layout:

```text
<data-root>/
  overview.csv
  images/
```

On the rented Ubuntu server, where the dataset is stored under `/root/autodl-tmp/data`, set:

```bash
export MAMTL_DATA_ROOT=/root/autodl-tmp/data
```

On Windows, the default data root is the parent directory of this repository (`C:\Data\MAMTL` for `C:\Data\MAMTL\repo`). The same environment variable can override it on either platform.

Check image filenames against `overview.csv` and write the fixed 70/15/15 split files with:

```bash
python prepare_data.py
```

The training scripts use the same class-stratified split when `--split-seed 2026` is selected. Leave-one-material-out folds construct their own train, validation, and held-out test partitions.

## Experiments A and B: Random Split

Experiment A trains the binary defect baseline. Experiment B adds a material-classification head to the shared backbone. Use the same split seed and model initialization seeds for both modes.

```bash
python run_experiments.py --mode defect --seed 42 --split-seed 2026 --out "$MAMTL_DATA_ROOT/runs/random/seed_42/defect"
python run_experiments.py --mode multitask --seed 42 --split-seed 2026 --out "$MAMTL_DATA_ROOT/runs/random/seed_42/multitask"
```

Repeat with additional matched seeds as needed. Each run saves the best checkpoint, training history, validation metrics, and test metrics as JSON. Summarize completed random-split runs with:

```bash
python summarize_results.py "$MAMTL_DATA_ROOT/runs/random" --out "$MAMTL_DATA_ROOT/results/random"
```

## Experiment C: Leave-One-Material-Out

Each fold excludes one material from both training and validation. The multitask material head predicts only the nine materials present during training; final test metrics use the binary defect head on the held-out material.

Run a one-epoch pilot on one material:

```bash
python cross_material.py --heldout-material glass --modes defect multitask --epochs 1 --seed 42
```

Run all ten folds for both models by omitting `--heldout-material`:

```bash
python cross_material.py --modes defect multitask --seed 42
```

Summarize completed folds with:

```bash
python summarize_cross_material.py "$MAMTL_DATA_ROOT/runs/cross_material" --out "$MAMTL_DATA_ROOT/results/cross_material"
```

The primary comparison is the paired defect-recognition performance of the baseline and multitask model on each held-out material. Material-supervised training is not assumed to improve this result.

Dataset reference: Buettner et al., "DefectAtlas-20: A real-world image dataset for joint material and surface-condition classification," *Data in Brief* (2026), DOI: `10.1016/j.dib.2026.113215`. The dataset is available from Zenodo under CC BY 4.0: DOI `10.5281/zenodo.21719370`.
