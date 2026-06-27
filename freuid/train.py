"""Train the FREUID baseline.

  python -m freuid.train --config configs/baseline.yaml
  python -m freuid.train --config configs/baseline.yaml --set train.epochs=10 model.name=convnext_tiny

Each run writes a self-contained folder under experiments/:
  config.yaml  meta.json  train.log  metrics.json  oof_val.csv  checkpoints/best.pt
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import dump_config, parse_args_and_config
from .data import load_train_df, make_loaders, make_split
from .losses import build_loss
from .metrics import freuid_score
from .model import build_model
from .utils import device_from_cfg, get_logger, git_hash, make_run_dir, set_seed


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logit = model(x).squeeze(1)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> None:
    cfg = parse_args_and_config()
    set_seed(cfg.seed)
    device = device_from_cfg(cfg)

    run_dir = make_run_dir(cfg.paths.exp_root, cfg.exp_name)
    log = get_logger("freuid", run_dir / "train.log")
    dump_config(cfg, run_dir / "config.yaml")

    df = load_train_df(cfg.paths.data_root)
    df = make_split(df, cfg)
    n_tr = int((~df["is_val"]).sum())
    n_va = int(df["is_val"].sum())
    n_pos = int(df.loc[~df["is_val"], "label"].sum())

    meta = {
        "git": git_hash(), "device": str(device), "exp_name": cfg.exp_name,
        "n_train": n_tr, "n_val": n_va, "val_scheme": cfg.val.scheme,
        "train_pos_rate": round(n_pos / max(1, n_tr), 4),
        "val_types": sorted(df.loc[df["is_val"], "type"].unique().tolist()),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info("run_dir=%s", run_dir)
    log.info("meta=%s", json.dumps(meta, ensure_ascii=False))

    tr_loader, va_loader, va_recap_loader, va_df = make_loaders(df, cfg)
    sel_key = "recap_freuid" if va_recap_loader is not None else "freuid"
    log.info("model selection by: %s", sel_key)
    model = build_model(cfg).to(device)

    pw = cfg.train.pos_weight
    if pw is None:
        pw = (n_tr - n_pos) / max(1, n_pos)
    criterion = build_loss(cfg, float(pw))
    log.info("loss: %s", cfg.train.get("loss") or "bce(pos_weight=%.3f)" % pw)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")

    steps_per_epoch = len(tr_loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = steps_per_epoch * cfg.train.warmup_epochs

    history, best = [], {"freuid": float("inf"), "recap_freuid": float("inf"), "epoch": -1}
    bad_epochs, gstep = 0, 0

    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        for it, (x, y) in enumerate(tr_loader):
            lr = cosine_warmup(gstep, total_steps, warmup_steps, cfg.train.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                logit = model(x).squeeze(1)
                loss = criterion(logit, y)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            gstep += 1
            if (it + 1) % cfg.train.log_every == 0:
                log.info("epoch %d it %d/%d loss %.4f lr %.2e",
                         epoch, it + 1, steps_per_epoch, running / (it + 1), lr)

        probs, labels = evaluate(model, va_loader, device)
        sc = freuid_score(labels, probs)
        sc["epoch"] = epoch
        sc["train_loss"] = running / steps_per_epoch
        if va_recap_loader is not None:
            rprobs, rlabels = evaluate(model, va_recap_loader, device)
            rsc = freuid_score(rlabels, rprobs)
            sc["recap_freuid"] = rsc["freuid"]
            sc["recap_audet"] = rsc["audet"]
            sc["recap_apcer@1%bpcer"] = rsc["apcer@1%bpcer"]
        history.append(sc)
        log.info("[val] epoch %d FREUID %.5f | recap %.5f",
                 epoch, sc["freuid"], sc.get("recap_freuid", float("nan")))

        if sc[sel_key] < best[sel_key]:
            best = {"epoch": epoch, **sc}
            torch.save({"model": model.state_dict(), "cfg": dict(cfg), "score": sc},
                       run_dir / "checkpoints" / "best.pt")
            va_df.assign(prob=probs).to_csv(run_dir / "oof_val.csv", index=False)
            bad_epochs = 0
            log.info("  -> new best %s %.5f (saved)", sel_key, best[sel_key])
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stop_patience:
                log.info("early stop at epoch %d (patience %d)", epoch, cfg.train.early_stop_patience)
                break

    (run_dir / "metrics.json").write_text(
        json.dumps({"best": best, "history": history}, indent=2, ensure_ascii=False))
    log.info("DONE best FREUID %.5f (recap %.5f, sel=%s) @ epoch %d | %s",
             best["freuid"], best.get("recap_freuid", float("nan")), sel_key, best["epoch"], run_dir)
    print(run_dir)  # stdout: run dir for downstream scripts


if __name__ == "__main__":
    main()
