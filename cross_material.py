import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_experiments import DefectDataset, Net, evaluate, load_rows, seed_all

def leave_one_material_out(rows, heldout, seed, val_fraction=0.15):
    test = [i for i, row in enumerate(rows) if row['material'] == heldout]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row['material'] != heldout:
            groups[row['class']].append(i)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(groups):
        ids = groups[label][:]
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_fraction))
        val.extend(ids[:n_val])
        train.extend(ids[n_val:])
    return train, val, test

def binary_counts(rows, indices):
    counts = {'defect': 0, 'no_defect': 0}
    for i in indices:
        counts[rows[i]['defect_label']] += 1
    return counts

def run_fold(args, rows, materials, classes, heldout, fold_index):
    split_seed = args.split_seed + fold_index
    train_ids, val_ids, test_ids = leave_one_material_out(rows, heldout, split_seed)
    seen_materials = sorted(set(r['material'] for r in rows if r['material'] != heldout))
    local_material_id = {name: i for i, name in enumerate(seen_materials)}
    fold_rows = []
    for row in rows:
        item = dict(row)
        item['material_i'] = local_material_id.get(row['material'], -1)
        fold_rows.append(item)

    fold_root = Path(args.out) / heldout
    fold_root.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        'held_out_material': heldout,
        'seen_materials': seen_materials,
        'split_seed': split_seed,
        'train_indices': train_ids,
        'val_indices': val_ids,
        'test_indices': test_ids,
        'train_defect_counts': binary_counts(rows, train_ids),
        'val_defect_counts': binary_counts(rows, val_ids),
        'test_defect_counts': binary_counts(rows, test_ids),
    }
    (fold_root / 'split.json').write_text(json.dumps(split_manifest, indent=2), encoding='utf-8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(DefectDataset(fold_rows, train_ids, args.images, True, args.size), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(DefectDataset(fold_rows, val_ids, args.images, False, args.size), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(DefectDataset(fold_rows, test_ids, args.images, False, args.size), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    for mode in args.modes:
        seed_all(args.seed)
        run_dir = fold_root / mode
        run_dir.mkdir(parents=True, exist_ok=True)
        net = Net(mode, len(seen_materials), len(classes), args.backbone, not args.no_pretrained).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        best = -1.0
        bad = 0
        history = []
        best_path = run_dir / 'best.pt'
        for epoch in range(1, args.epochs + 1):
            net.train()
            total_loss = 0.0
            start = time.time()
            for x, material, defect, _ in train_loader:
                x = x.to(device, non_blocking=True)
                material = material.to(device)
                defect = defect.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    defect_logits, material_logits, _ = net(x)
                    loss = F.cross_entropy(defect_logits, defect)
                    if mode == 'multitask':
                        loss = loss + args.lambda_material * F.cross_entropy(material_logits, material)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            val = evaluate(net, val_loader, device, mode)
            score = val['macro_f1']
            history.append({'epoch': epoch, 'train_loss': total_loss / len(train_loader), 'val': val, 'seconds': time.time() - start})
            print(f'{heldout} {mode} epoch {epoch:02d} loss {total_loss / len(train_loader):.4f} val_f1 {score:.4f}', flush=True)
            if score > best:
                best, bad = score, 0
                torch.save(net.state_dict(), best_path)
            else:
                bad += 1
            if bad >= args.patience:
                break

        net.load_state_dict(torch.load(best_path, map_location=device))
        test = evaluate(net, test_loader, device, mode, include_material_metrics=False)
        result = {
            'mode': mode,
            'seed': args.seed,
            'split_seed': split_seed,
            'backbone': args.backbone,
            'held_out_material': heldout,
            'seen_materials': seen_materials,
            'material_head_classes': seen_materials if mode == 'multitask' else [],
            'train_size': len(train_ids),
            'val_size': len(val_ids),
            'test_size': len(test_ids),
            'test_defect_counts': split_manifest['test_defect_counts'],
            'best_val_macro_f1': best,
            'test': test,
            'history': history,
            'num_params': sum(p.numel() for p in net.parameters()),
        }
        (run_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'{heldout} {mode} test_f1 {test["macro_f1"]:.4f} auroc {test["defect_auroc"]:.4f}', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=r'C:\Data\MAMTL\overview.csv')
    parser.add_argument('--images', default=r'C:\Data\MAMTL\images')
    parser.add_argument('--out', default=r'C:\Data\MAMTL\runs\cross_material')
    parser.add_argument('--heldout-material', action='append', default=None)
    parser.add_argument('--modes', nargs='+', choices=['defect', 'multitask'], default=['defect', 'multitask'])
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-material', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    rows, materials, classes = load_rows(args.csv)
    heldout = args.heldout_material or materials
    unknown = sorted(set(heldout) - set(materials))
    if unknown:
        raise SystemExit(f'Unknown held-out material(s): {unknown}; available: {materials}')
    for fold_index, material in enumerate(heldout):
        run_fold(args, rows, materials, classes, material, materials.index(material))

if __name__ == '__main__':
    main()
