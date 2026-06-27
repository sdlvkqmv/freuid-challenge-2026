"""Test-time adaptation (direction G) — re-align a trained model's BN stats to the
unlabeled test (recapture) domain without retraining. Bolts onto an existing run's best.pt.

Two modes:
  adabn : recompute BN running mean/var over the whole test set (parameter-free, AdaBN).
  tent  : adabn warm-start + entropy-minimize the BN affine params (gamma/beta) on test
          batches (Tent, Wang et al. 2021). Only BN affine trainable; everything else frozen.

Rationale (docs/SUMMARY root finding #0): train is ~all digital, test emphasizes
print-and-capture. The 06 model collapses on the recapture domain partly because its BN
statistics are calibrated to the digital train distribution. Adapting BN to the test
distribution is the cheapest possible domain-gap fix — no labels, no retraining.

  python -m freuid.tta --run experiments/attempt06_... --mode adabn --gpu 6
  python -m freuid.tta --run experiments/attempt06_... --mode tent --lr 1e-3 --steps 1 --gpu 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import load_config
from .data import list_available_test_ids, make_test_loader
from .model import build_model
from .utils import device_from_cfg, get_logger


def _bn_modules(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def reset_bn_for_adabn(model: nn.Module) -> None:
    """Reset BN running stats and switch to cumulative-average accumulation (momentum=None)."""
    for m in _bn_modules(model):
        m.reset_running_stats()
        m.momentum = None          # cumulative moving average over all batches seen
        m.train()                  # use+update batch stats


@torch.no_grad()
def adabn_collect(model, loader, device) -> None:
    """One pass over the test set to accumulate BN stats on the test distribution."""
    model.eval()                   # freeze dropout etc.
    reset_bn_for_adabn(model)      # but BN layers back in train mode
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            model(x)


def configure_tent(model: nn.Module) -> list[nn.Parameter]:
    """Freeze everything except BN affine params; BN uses batch stats (train mode)."""
    model.eval()
    params = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
            m.requires_grad_(False)
            if m.weight is not None:
                m.weight.requires_grad_(True); params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True); params.append(m.bias)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    return params


def _bin_entropy(logit: torch.Tensor) -> torch.Tensor:
    """Mean binary entropy of sigmoid(logit). logit: (B,)."""
    p = torch.sigmoid(logit)
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


@torch.no_grad()
def infer(model, loader, device) -> tuple[list[str], np.ndarray]:
    model.eval()
    # keep BN in whatever mode the caller set (adabn->eval stats now frozen; tent->batch stats)
    for m in _bn_modules(model):
        if m.momentum is None:
            m.eval()               # adabn: use the accumulated running stats
    ids, probs = [], []
    for x, batch_ids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logit = model(x).squeeze(1)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        ids.extend(batch_ids)
    return ids, np.concatenate(probs)


def tent_adapt_and_infer(model, loader, device, lr: float, steps: int, log) -> tuple[list[str], np.ndarray]:
    """Online Tent: per batch, take `steps` entropy-min updates on BN affine, then record preds."""
    params = configure_tent(model)
    opt = torch.optim.Adam(params, lr=lr)
    ids, probs = [], []
    for bi, (x, batch_ids) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        for _ in range(steps):
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logit = model(x).squeeze(1)
            loss = _bin_entropy(logit.float())
            loss.backward()
            opt.step()
        with torch.no_grad():
            logit = model(x).squeeze(1)
        probs.append(torch.sigmoid(logit).float().detach().cpu().numpy())
        ids.extend(batch_ids)
        if bi % 50 == 0:
            log.info("tent batch %d loss %.4f", bi, float(loss))
    return ids, np.concatenate(probs)


def write_submission(cfg, ids, probs, out: Path, log) -> None:
    pred = dict(zip(ids, probs.astype(float)))
    sub = pd.read_csv(Path(cfg.paths.data_root) / "sample_submission.csv")
    fill = float(cfg.predict.fill_value)
    sub["label"] = sub["id"].map(pred).fillna(fill)
    n_real = sub["id"].isin(pred).sum()
    log.info("submission rows %d | real preds %d | filled %d (=%.2f)",
             len(sub), n_real, len(sub) - n_real, fill)
    sub.to_csv(out, index=False)
    log.info("wrote %s", out)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=["adabn", "tent"], required=True)
    ap.add_argument("--lr", type=float, default=1e-3, help="tent only")
    ap.add_argument("--steps", type=int, default=1, help="tent updates per batch")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None, help="override batch size (tent needs smaller for backprop)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(str(run / "config.yaml"))
    if args.gpu is not None:
        cfg.gpu = args.gpu
    if args.bs is not None:
        cfg.data.batch_size = args.bs
    log = get_logger("tta")
    device = device_from_cfg(cfg)

    ckpt = torch.load(run / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    log.info("loaded %s (val FREUID %.5f) mode=%s", run / "checkpoints/best.pt",
             ckpt["score"]["freuid"], args.mode)

    test_ids = list_available_test_ids(cfg.paths.data_root)
    log.info("public_test images available: %d", len(test_ids))
    loader = make_test_loader(test_ids, cfg)

    if args.mode == "adabn":
        adabn_collect(model, loader, device)
        ids, probs = infer(model, loader, device)
        tag = "adabn"
    else:
        # Tent benefits from an AdaBN warm-start of running stats, but Tent itself uses
        # batch stats; we go straight to online tent (configure_tent sets BN train mode).
        ids, probs = tent_adapt_and_infer(model, loader, device, args.lr, args.steps, log)
        tag = f"tent_lr{args.lr}_s{args.steps}"

    out = Path(args.out) if args.out else run / f"submission_{tag}.csv"
    write_submission(cfg, ids, probs, out, log)


if __name__ == "__main__":
    main()
