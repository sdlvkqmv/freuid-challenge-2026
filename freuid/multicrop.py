"""Multi-crop max-aggregation TTA (direction E / ROI-lite) — bolt onto a trained run.

Tampered regions (physical manipulation, GenAI face/field edits) are spatially TINY; a
single global-pooled forward dilutes a small high-evidence region. Score several spatial
crops independently and aggregate by max (or top-k mean): a crop landing on the tampered
region keeps its high fraud score instead of being averaged away. This is a detector-free
first test of the spatial-focus hypothesis before building a real face/text ROI pipeline.

Crops: the full-resize view (= the base model's normal input) plus a grid of overlapping
tiles at `crop_scale` of the short side. 06 was trained with RandomResizedCrop 0.7-1.0, so
tiles are kept near that scale to stay in-distribution.

  python -m freuid.multicrop --run experiments/attempt06_... --grid 2 --crop-scale 0.7 \
      --agg max --gpu 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import load_config
from .data import IMAGENET_MEAN, IMAGENET_STD, img_path, list_available_test_ids
from .model import build_model
from .utils import device_from_cfg, get_logger


def _crops(img: Image.Image, size: int, grid: int, scale: float, mean, std) -> torch.Tensor:
    """Return a (K,3,size,size) tensor: full-resize view + grid*grid overlapping tiles."""
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    W, H = img.size
    views = [img.resize((size, size), Image.BILINEAR)]               # global view
    cw, ch = int(W * scale), int(H * scale)
    if grid > 1 and cw > 0 and ch > 0:
        xs = np.linspace(0, W - cw, grid).astype(int)
        ys = np.linspace(0, H - ch, grid).astype(int)
        for y in ys:
            for x in xs:
                views.append(img.crop((x, y, x + cw, y + ch)).resize((size, size), Image.BILINEAR))
    return torch.stack([norm(v) for v in views])                     # (K,3,size,size)


@torch.no_grad()
def infer(model, test_ids, cfg, device, grid, scale, agg, topk, log):
    model.eval()
    size = cfg.data.img_size
    mean = tuple(cfg.data.get("mean") or IMAGENET_MEAN)
    std = tuple(cfg.data.get("std") or IMAGENET_STD)
    root = cfg.paths.data_root
    bs = max(1, cfg.data.batch_size)
    ids, probs = [], []
    for i, img_id in enumerate(test_ids):
        img = Image.open(img_path(root, "test", img_id)).convert("RGB")
        x = _crops(img, size, grid, scale, mean, std).to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).squeeze(1)).float()           # (K,)
        if agg == "max":
            score = p.max()
        elif agg == "topk":
            score = p.topk(min(topk, p.numel())).values.mean()
        elif agg == "mean":
            score = p.mean()
        else:
            raise ValueError(agg)
        ids.append(img_id); probs.append(float(score))
        if (i + 1) % 1000 == 0:
            log.info("scored %d/%d", i + 1, len(test_ids))
    return ids, np.array(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--grid", type=int, default=2, help="grid x grid tiles (+1 global view)")
    ap.add_argument("--crop-scale", type=float, default=0.7)
    ap.add_argument("--agg", choices=["max", "topk", "mean"], default="max")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(str(run / "config.yaml"))
    if args.gpu is not None:
        cfg.gpu = args.gpu
    log = get_logger("multicrop")
    device = device_from_cfg(cfg)

    ckpt = torch.load(run / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    log.info("loaded %s | grid=%d scale=%.2f agg=%s", run / "checkpoints/best.pt",
             args.grid, args.crop_scale, args.agg)

    test_ids = list_available_test_ids(cfg.paths.data_root)
    log.info("public_test images: %d", len(test_ids))
    ids, probs = infer(model, test_ids, cfg, device, args.grid, args.crop_scale, args.agg, args.topk, log)

    pred = dict(zip(ids, probs.astype(float)))
    sub = pd.read_csv(Path(cfg.paths.data_root) / "sample_submission.csv")
    fill = float(cfg.predict.fill_value)
    sub["label"] = sub["id"].map(pred).fillna(fill)
    n_real = sub["id"].isin(pred).sum()
    log.info("rows %d | real %d | filled %d", len(sub), n_real, len(sub) - n_real)
    tag = f"mc_g{args.grid}_s{args.crop_scale}_{args.agg}"
    out = Path(args.out) if args.out else run / f"submission_{tag}.csv"
    sub.to_csv(out, index=False)
    log.info("wrote %s", out)
    print(out)


if __name__ == "__main__":
    main()
