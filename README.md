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
