import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "precision_macro", "recall_macro", "defect_auroc")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=r"C:\Data\MAMTL\runs")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    material_scores = []
    for path in sorted(root.glob("*_seed*/*_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        test = result["test"]
        runs.append({"mode": result["mode"], "seed": result["seed"], **{m: test.get(m) for m in METRICS}})
        names = result.get("material_names", [])
        split_manifest = path.parent / "splits.json"
        if not names and split_manifest.exists():
            names = json.loads(split_manifest.read_text(encoding="utf-8")).get("materials", [])
        for index, score in test.get("defect_f1_by_material", {}).items():
            i = int(index)
            material_scores.append({"mode": result["mode"], "seed": result["seed"], "material": names[i] if i < len(names) else index, **score})

    if not runs:
        raise SystemExit(f"No completed result files found under {root}")
    fields = ["mode", "seed", *METRICS]
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(runs)

    groups = {}
    for row in runs:
        groups.setdefault(row["mode"], []).append(row)
    with (out / "mean_std.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "n", *[v for m in METRICS for v in (m + "_mean", m + "_std")]])
        writer.writeheader()
        for mode, group in sorted(groups.items()):
            row = {"mode": mode, "n": len(group)}
            for metric in METRICS:
                values = [r[metric] for r in group if r[metric] is not None]
                row[metric + "_mean"] = statistics.mean(values) if values else ""
                row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else (0 if values else "")
            writer.writerow(row)
    with (out / "per_material.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["mode", "seed", "material", "macro_f1", "count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(material_scores)
    print(f"Wrote summaries to {out.resolve()}")

if __name__ == "__main__":
    main()
