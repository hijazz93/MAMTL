# MAMTL experiment repository

Material-Aware Multi-Task Learning for real-world surface defect recognition.

## Data layout

Keep the DefectAtlas-20 data outside Git:

- `C:\Data\MAMTL\overview.csv`
- `C:\Data\MAMTL\images\`
- `C:\Data\MAMTL\splits\`

Prepare fixed splits with:

```powershell
python prepare_data.py
```

## Experiments

```powershell
python run_experiments.py --csv C:\Data\MAMTL\overview.csv --images C:\Data\MAMTL\images --out C:\Data\MAMTL\runs\defect --mode defect
python run_experiments.py --csv C:\Data\MAMTL\overview.csv --images C:\Data\MAMTL\images --out C:\Data\MAMTL\runs\multitask --mode multitask
python run_experiments.py --csv C:\Data\MAMTL\overview.csv --images C:\Data\MAMTL\images --out C:\Data\MAMTL\runs\fusion --mode fusion
```

Each run saves the best checkpoint, training history, validation metrics, and test metrics as JSON.

The comparison runs should use the same fixed split (`--split-seed 2026`). Vary `--seed` for independent model initializations, and compare the binary defect baseline (`defect`), shared multi-task model (`multitask`), and material-probability fusion model (`fusion`). Main binary metrics include Macro-F1, balanced accuracy, precision, recall, AUROC, and per-material Macro-F1.

Summarize completed runs with:

```powershell
python summarize_results.py C:\Data\MAMTL\runs --out results
```

## Leave-One-Material-Out Evaluation

This evaluates defect recognition on one material entirely excluded from training and validation. The multi-task material head is sized to the nine seen materials; only the binary defect head is scored on the held-out material. By default the script runs all ten folds for the defect baseline and multi-task model. Run a single pilot fold with:

```powershell
python cross_material.py --heldout-material glass --modes defect multitask --epochs 1
```

Run the full protocol by omitting `--heldout-material`, then summarize with:

```powershell
python summarize_cross_material.py C:\Data\MAMTL\runs\cross_material --out cross_material_results
```

For each held-out material, train and validation samples come only from the other nine materials. The held-out material is used only for the final defect test metrics.

Dataset reference: Buettner et al., "DefectAtlas-20: A real-world image dataset for joint material and surface-condition classification," *Data in Brief* (2026), DOI: `10.1016/j.dib.2026.113215`. The dataset is available from Zenodo under CC BY 4.0: DOI `10.5281/zenodo.21719370`.
