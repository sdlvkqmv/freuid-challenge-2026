"""Final-submission training script (attempt31 "MIL field-crop v2").

MIL field-crop EfficientNet-B3+SRM, 320px crops, 6 epochs, hflip aug+TTA,
static-zone crops included. Document score = MAX over K=13 per-type field
crops; per-crop hflip-TTA at val/inference (average of 2 views).

Verbatim copy of the script archived in the winning run directory, with only
paths rewritten repo-relative (config, field boxes, test-type assignment).
Usage: python scripts/train_mil_v2.py <GPU>
Test-set document types (assets/test_type_assignment.csv) come from dhash
nearest-neighbour matching against training images — see
submission/prepare_submission.py, which recomputes them from scratch.
"""
import sys, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

REPO=str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
from freuid.config import load_config
from freuid.data import make_split, load_train_df
from freuid.metrics import freuid_score
from freuid.model import build_model
from freuid.utils import device_from_cfg, get_logger, make_run_dir, set_seed

GPU=int(sys.argv[1]) if len(sys.argv)>1 else 5
EPOCHS=6; CROP=320; KMAX=13; BATCH=3; LR=3e-4; WD=0.05
CANON={"BENIN/DL":(1000,1585),"EGYPT/DL":(875,1387),"GUINEA/DL":(1000,1584),
       "MAURITIUS/ID":(1000,1585),"MOZAMBIQUE/DL":(630,1000)}
BOXES=json.loads((Path(REPO)/"assets/field_boxes_v2.json").read_text())
IMEAN=(0.485,0.456,0.406); ISTD=(0.229,0.224,0.225)
SUB={"train":"train/train","test":"public_test/public_test"}; DATA=Path(REPO)/"data/extracted"

def crop_pil(idv, split, typ):
    im=Image.open(DATA/SUB[split]/f"{idv}.jpeg").convert("RGB")
    W,H=CANON[typ]
    if im.size!=(W,H): im=im.resize((W,H))
    cs=[im.crop((x0,y0,x1,y1)).resize((CROP,CROP)) for y0,y1,x0,x1 in BOXES[typ]]
    while len(cs)<KMAX: cs.append(cs[-1])
    return cs[:KMAX]

def make_tfm(train):
    ops=[]
    if train: ops+=[transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(0.1,0.1,0.1)]
    ops+=[transforms.ToTensor(), transforms.Normalize(IMEAN,ISTD)]
    return transforms.Compose(ops)

class MILSet(Dataset):
    def __init__(self, df, split, train, has_label=True):
        self.ids=df["id"].tolist(); self.types=df["type"].tolist()
        self.labels=df["label"].tolist() if has_label else None
        self.split=split; self.tfm=make_tfm(train); self.has_label=has_label
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        x=torch.stack([self.tfm(c) for c in crop_pil(self.ids[i],self.split,self.types[i])])
        if self.has_label: return x, torch.tensor(float(self.labels[i]))
        return x, self.ids[i]

class MIL(nn.Module):
    def __init__(self,cfg): super().__init__(); self.scorer=build_model(cfg)
    def forward(self,x):                 # x (B,K,3,H,W) -> per-crop logits (B,K)
        B,K=x.shape[:2]
        return self.scorer(x.reshape(B*K,*x.shape[2:])).squeeze(1).view(B,K)

def doc_logit(model,x,tta=False):
    l=model(x)
    if tta: l=0.5*(l+model(torch.flip(x,dims=[-1])))
    return l.max(dim=1).values

def cosw(s,t,w,b):
    if s<w: return b*(s+1)/max(1,w)
    p=(s-w)/max(1,t-w); return 0.5*b*(1+math.cos(math.pi*p))

@torch.no_grad()
def evaluate(model,loader,device,tta=True):
    model.eval(); ps=[]; ys=[]
    for x,y in loader:
        x=x.to(device,non_blocking=True)
        with torch.autocast("cuda",enabled=device.type=="cuda"):
            d=doc_logit(model,x,tta)
        ps.append(torch.sigmoid(d).float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)

def main():
    cfg=load_config(REPO+"/configs/mil_fieldcrop_v2.yaml")
    cfg.set_dotted("model.name","tf_efficientnet_b3"); cfg.set_dotted("model.streams",["rgb","srm"])
    cfg.set_dotted("model.pretrained",True); cfg.set_dotted("paths.exp_root",REPO+"/experiments")
    cfg.set_dotted("exp_name","attempt31_mil_fieldcrop_v2_320px_6ep_hflip_static")
    cfg.set_dotted("gpu",GPU); cfg.set_dotted("device","cuda")
    set_seed(42); device=device_from_cfg(cfg)
    run=make_run_dir(cfg.paths.exp_root, cfg.exp_name); log=get_logger("milv2", run/"train.log")
    (run/"config.yaml").write_text((Path(REPO)/"configs/mil_fieldcrop_v2.yaml").read_text())
    Path(run/"stage2_mil_v2.py").write_text(Path(__file__).read_text())
    (run/"field_boxes_v2.json").write_text(json.dumps(BOXES))
    df=load_train_df(cfg.paths.data_root); df=make_split(df,cfg)
    tr=df[~df["is_val"]].reset_index(drop=True); va=df[df["is_val"]].reset_index(drop=True)
    n=len(tr); npos=int(tr["label"].sum()); pw=(n-npos)/max(1,npos)
    log.info("train %d(pos %d) val %d | K=%d CROP=%d BATCH=%d EP=%d LR=%.1e pw=%.3f hflip+TTA+static",
             n,npos,len(va),KMAX,CROP,BATCH,EPOCHS,LR,pw)
    common=dict(num_workers=12,pin_memory=True,persistent_workers=True)
    trl=DataLoader(MILSet(tr,"train",True),batch_size=BATCH,shuffle=True,drop_last=True,**common)
    val=DataLoader(MILSet(va,"train",False),batch_size=BATCH,shuffle=False,**common)
    model=MIL(cfg).to(device)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    spe=len(trl); total=spe*EPOCHS; warm=spe
    best={"freuid":float("inf"),"epoch":-1}; g=0; history=[]
    for ep in range(EPOCHS):
        model.train(); rl=0.0; t0=time.time()
        for it,(x,y) in enumerate(trl):
            lr=cosw(g,total,warm,LR)
            for pgp in opt.param_groups: pgp["lr"]=lr
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",enabled=True):
                loss=crit(doc_logit(model,x,False),y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            rl+=loss.item(); g+=1
            if (it+1)%400==0: log.info("ep %d it %d/%d loss %.4f lr %.2e (%.0fs)",ep,it+1,spe,rl/(it+1),lr,time.time()-t0)
        p,yv=evaluate(model,val,device,tta=False); sc=freuid_score(yv,p); sc["epoch"]=ep; sc["train_loss"]=rl/spe
        history.append(sc); log.info("[val] ep %d doc-FREUID %.5f (audet %.5f apcer %.5f)",ep,sc["freuid"],sc["audet"],sc["apcer@1%bpcer"])
        # keep last-epoch ckpt always; keep best-val separately
        torch.save({"model":model.state_dict(),"score":sc},run/"checkpoints"/"last.pt")
        if sc["freuid"]<best["freuid"]:
            best={"epoch":ep,**sc}
            torch.save({"model":model.state_dict(),"score":sc},run/"checkpoints"/"best.pt")
            va.assign(prob=p).to_csv(run/"oof_val.csv",index=False); log.info("  -> new best %.5f",best["freuid"])
    (run/"metrics.json").write_text(json.dumps({"best":best,"history":history},indent=2))
    log.info("TRAIN DONE best doc-FREUID %.5f @ep %d",best["freuid"],best["epoch"])
    # inference with BEST-val ckpt (submit best per team-lead)
    tt=pd.read_csv(REPO+"/assets/test_type_assignment.csv")[["id","type"]]
    ck=torch.load(run/"checkpoints"/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"]); model.eval()
    tl=DataLoader(MILSet(tt,"test",False,has_label=False),batch_size=BATCH,shuffle=False,**common)
    ids=[]; scs=[]
    with torch.no_grad():
        for x,bid in tl:
            x=x.to(device,non_blocking=True)
            with torch.autocast("cuda",enabled=True): d=doc_logit(model,x,tta=True)
            scs.append(torch.sigmoid(d).float().cpu().numpy()); ids.extend(bid)
    pred=dict(zip(ids,np.concatenate(scs).astype(float)))
    sub=pd.read_csv(cfg.paths.data_root+"/sample_submission.csv"); sub["label"]=sub["id"].map(pred).fillna(0.5)
    sub.to_csv(run/"submission.csv",index=False)
    log.info("wrote submission rows %d real %d @ %s",len(sub),sub["id"].isin(pred).sum(),run/"submission.csv")
    print(run)

if __name__=="__main__": main()
