import argparse, csv, json, os, random, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_rows(csv_path):
    with open(csv_path, encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    mats = sorted({r['material'] for r in rows}); mat_to_i = {x:i for i,x in enumerate(mats)}
    classes = sorted({r['class'] for r in rows}); cls_to_i = {x:i for i,x in enumerate(classes)}
    for r in rows:
        r['material_i'] = mat_to_i[r['material']]; r['defect_i'] = int(r['defect_label'] == 'defect'); r['class_i'] = cls_to_i[r['class']]
    return rows, mats, classes

def stratified_split(rows, seed, train_frac=.70, val_frac=.15):
    rng = random.Random(seed); groups = defaultdict(list)
    for i,r in enumerate(rows): groups[r['class_i']].append(i)
    tr=[]; va=[]; te=[]
    for _, inds in sorted(groups.items()):
        rng.shuffle(inds); n=len(inds); nt=round(n*train_frac); nv=round(n*val_frac)
        tr += inds[:nt]; va += inds[nt:nt+nv]; te += inds[nt+nv:]
    return tr, va, te

class DefectDataset(Dataset):
    def __init__(self, rows, indices, image_root, train=False, size=224):
        self.rows=[rows[i] for i in indices]; self.image_root=Path(image_root)
        self.image_paths={p.name:p for p in self.image_root.rglob('*.png')}
        norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        self.tf=transforms.Compose(([transforms.RandomResizedCrop(size,scale=(.75,1.0)), transforms.RandomHorizontalFlip()] if train else [transforms.Resize((size,size))]) + [transforms.ToTensor(),norm])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]; p=self.image_paths.get(r['sample_name'])
        if p is None: raise FileNotFoundError(str(self.image_root / r['sample_name']))
        with Image.open(p) as im: x=self.tf(im.convert('RGB'))
        return x, torch.tensor(r['material_i']), torch.tensor(r['defect_i']), torch.tensor(r['class_i'])

class Net(nn.Module):
    def __init__(self, mode, nmat, nclass, backbone='resnet50', pretrained=True):
        super().__init__(); self.mode=mode
        ctor=getattr(models, backbone)
        weights = 'DEFAULT' if pretrained else None
        try: base=ctor(weights=weights)
        except Exception as e:
            print('Pretrained weights unavailable, using random init:', e); base=ctor(weights=None)
        dim=base.fc.in_features; base.fc=nn.Identity(); self.backbone=base
        self.material=nn.Linear(dim,nmat); self.defect=nn.Linear(dim,nclass if mode=='joint20' else 2)
        self.fusion=nn.Linear(dim+nmat,2) if mode=='fusion' else None
    def forward(self,x):
        f=self.backbone(x); m=self.material(f)
        if self.mode=='joint20': return self.defect(f), m, None
        if self.mode=='defect': return self.defect(f), None, None
        if self.mode=='multitask': return self.defect(f), m, None
        p=torch.softmax(m,1); return self.fusion(torch.cat([f,p],1)), m, p

def metrics(y, p, ncls=2):
    out={'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'precision_macro':precision_score(y,p,average='macro',zero_division=0),'recall_macro':recall_score(y,p,average='macro',zero_division=0),'confusion_matrix':confusion_matrix(y,p).tolist()}
    return out

@torch.no_grad()
def evaluate(net, loader, device, mode, include_material_metrics=True):
    net.eval(); ys=[]; ps=[]; probs=[]; mats=[]; matp=[]; materials=[]; losses=[]; ce=nn.CrossEntropyLoss()
    for x,m,d,c in loader:
        x=x.to(device,non_blocking=True); m=m.to(device); d=d.to(device); c=c.to(device)
        out, mo, _=net(x); target=c if mode=='joint20' else d; losses.append(ce(out,target).item()); ps.extend(out.argmax(1).cpu().tolist()); ys.extend(target.cpu().tolist())
        if mode!='joint20':
            probs.extend(out.softmax(1)[:,1].cpu().tolist())
            if include_material_metrics: materials.extend(m.cpu().tolist())
        if mo is not None and include_material_metrics: mats.extend(m.cpu().tolist()); matp.extend(mo.argmax(1).cpu().tolist())
    z=metrics(ys,ps,20 if mode=='joint20' else 2); z['loss']=float(np.mean(losses))
    if mode!='joint20':
        try: z['defect_auroc']=roc_auc_score(ys,probs)
        except ValueError: z['defect_auroc']=None
        if include_material_metrics:
            z['defect_f1_by_material']={}
            for mid in sorted(set(materials)):
                keep=[i for i,v in enumerate(materials) if v==mid]
                z['defect_f1_by_material'][str(mid)]={'macro_f1':f1_score([ys[i] for i in keep],[ps[i] for i in keep],average='macro',zero_division=0),'count':len(keep)}
    if mats: z['material_macro_f1']=f1_score(mats,matp,average='macro',zero_division=0); z['material_accuracy']=accuracy_score(mats,matp)
    return z

def train(args):
    seed_all(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device)
    rows,mats,classes=load_rows(args.csv); splits=stratified_split(rows,args.split_seed)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); json.dump({'train':splits[0],'val':splits[1],'test':splits[2],'materials':mats,'classes':classes},open(out/'splits.json','w'))
    image_root=args.images
    tr=DataLoader(DefectDataset(rows,splits[0],image_root,True,args.size),batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True)
    va=DataLoader(DefectDataset(rows,splits[1],image_root,False,args.size),batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    te=DataLoader(DefectDataset(rows,splits[2],image_root,False,args.size),batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    net=Net(args.mode,len(mats),len(classes),args.backbone,not args.no_pretrained).to(device); opt=torch.optim.AdamW(net.parameters(),lr=args.lr); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); best=-1; bad=0; best_path=out/f'{args.mode}_best.pt'; hist=[]
    for ep in range(1,args.epochs+1):
        net.train(); t=time.time(); run=0
        for x,m,d,c in tr:
            x=x.to(device,non_blocking=True); m=m.to(device); d=d.to(device); c=c.to(device); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                out1, outm, _=net(x); target=c if args.mode=='joint20' else d; loss=nn.functional.cross_entropy(out1,target)
                if args.mode in ('multitask','fusion'): loss=loss+args.lambda_material*nn.functional.cross_entropy(outm,m)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run+=loss.item()
        vm=evaluate(net,va,device,args.mode); score=vm['macro_f1']; hist.append({'epoch':ep,'train_loss':run/len(tr),'val':vm,'seconds':time.time()-t}); print(f'epoch {ep:02d} loss {run/len(tr):.4f} val_f1 {score:.4f} val_acc {vm["accuracy"]:.4f}',flush=True)
        if score>best: best=score; bad=0; torch.save(net.state_dict(),best_path)
        else: bad+=1
        if bad>=args.patience: break
    net.load_state_dict(torch.load(best_path,map_location=device)); result={'mode':args.mode,'seed':args.seed,'split_seed':args.split_seed,'backbone':args.backbone,'input_size':args.size,'learning_rate':args.lr,'lambda_material':args.lambda_material,'best_val_macro_f1':best,'val':evaluate(net,va,device,args.mode),'test':evaluate(net,te,device,args.mode),'history':hist,'num_params':sum(p.numel() for p in net.parameters())}
    result['material_names']=mats
    json.dump(result,open(out/f'{args.mode}_result.json','w'),indent=2); print(json.dumps(result['test'],indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',default='overview.csv'); ap.add_argument('--images',default='images'); ap.add_argument('--out',default='runs'); ap.add_argument('--mode',choices=['defect','multitask','fusion','joint20'],required=True); ap.add_argument('--backbone',default='resnet50'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--patience',type=int,default=7); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--size',type=int,default=224); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--lambda-material',type=float,default=1.0); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--split-seed',type=int,default=2026); ap.add_argument('--no-pretrained',action='store_true'); train(ap.parse_args())
