from collections import defaultdict
from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "images"
CSV_PATH = ROOT / "overview.csv"
SPLIT_ROOT = ROOT / "splits"
SEED = 2026

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

image_index = defaultdict(list)
for path in IMAGE_ROOT.rglob("*.png"):
    image_index[path.name].append(path)

csv_names = {row["sample_name"] for row in rows}
missing = sorted(name for name in csv_names if name not in image_index)
duplicates = sorted((name, paths) for name, paths in image_index.items() if len(paths) > 1)
extra = sorted(path for name, paths in image_index.items() if name not in csv_names for path in paths)

print(f"CSV rows: {len(rows)}")
print(f"PNG files: {sum(len(paths) for paths in image_index.values())}")
print(f"Missing images: {len(missing)}")
print(f"Duplicate filenames: {len(duplicates)}")
print(f"Extra images: {len(extra)}")

if missing:
    (ROOT / "missing_images.txt").write_text("\n".join(missing), encoding="utf-8")
    raise RuntimeError("Missing images; see missing_images.txt")
if duplicates:
    raise RuntimeError("Duplicate image filenames found")

rng = random.Random(SEED)
groups = defaultdict(list)
for row in rows:
    groups[row["class"]].append(row)

train, val, test = [], [], []
for class_name in sorted(groups):
    group = groups[class_name][:]
    rng.shuffle(group)
    n_train = round(len(group) * 0.70)
    n_val = round(len(group) * 0.15)
    train.extend(group[:n_train])
    val.extend(group[n_train:n_train + n_val])
    test.extend(group[n_train + n_val:])

SPLIT_ROOT.mkdir(exist_ok=True)
fieldnames = list(rows[0].keys()) + ["split"]
for name, data in (("train", train), ("val", val), ("test", test)):
    with (SPLIT_ROOT / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            output = dict(row)
            output["split"] = name
            writer.writerow(output)

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")
print("Class counts (train/val/test):")
for class_name in sorted(groups):
    counts = [sum(row["class"] == class_name for row in data) for data in (train, val, test)]
    print(class_name, *counts)
