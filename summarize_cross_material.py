import argparse
import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('MAMTL_DATA_ROOT', ROOT.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default=str(DATA_ROOT / 'runs' / 'cross_material'))
    parser.add_argument('--out', default='cross_material_results')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.glob('*/*/result.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        metrics = item['test']
        rows.append({'material': item['held_out_material'], 'mode': item['mode'], 'test_size': item['test_size'], 'defect_count': item['test_defect_counts']['defect'], 'no_defect_count': item['test_defect_counts']['no_defect'], 'macro_f1': metrics['macro_f1'], 'balanced_accuracy': metrics['balanced_accuracy'], 'accuracy': metrics['accuracy'], 'auroc': metrics['defect_auroc']})
    if not rows:
        raise SystemExit(f'No leave-one-material-out results found under {root}')
    fields = list(rows[0])
    with (out / 'per_material.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows: groups.setdefault(row['mode'], []).append(row)
    metrics = ('macro_f1', 'balanced_accuracy', 'accuracy', 'auroc')
    with (out / 'mean_std.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['mode', 'folds', *[k for m in metrics for k in (m + '_mean', m + '_std')]])
        writer.writeheader()
        for mode, group in sorted(groups.items()):
            result = {'mode': mode, 'folds': len(group)}
            for metric in metrics:
                values = [r[metric] for r in group if r[metric] is not None]
                result[metric + '_mean'] = statistics.mean(values) if values else ''
                result[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else (0 if values else '')
            writer.writerow(result)
    materials = sorted({r['material'] for r in rows})
    lookup = {(r['material'], r['mode']): r for r in rows}
    deltas = []
    for material in materials:
        if (material, 'defect') in lookup and (material, 'multitask') in lookup:
            base = lookup[(material, 'defect')]['macro_f1']
            mtl = lookup[(material, 'multitask')]['macro_f1']
            deltas.append({'material': material, 'baseline_macro_f1': base, 'multitask_macro_f1': mtl, 'delta': mtl - base})
    with (out / 'paired_delta.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['material', 'baseline_macro_f1', 'multitask_macro_f1', 'delta']); writer.writeheader(); writer.writerows(deltas)
    print(f'Wrote leave-one-material-out summaries to {out.resolve()}')

if __name__ == '__main__':
    main()
