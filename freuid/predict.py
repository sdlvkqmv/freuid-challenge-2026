"""Inference -> Kaggle submission.

  python -m freuid.predict --config configs/baseline.yaml --set _run=experiments/baseline_effb0_XXXX

Predicts on every available public_test image, then aligns to sample_submission.csv.
Test ids without an image yet (full test set not released) are filled with predict.fill_value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import list_available_test_ids, make_test_loader
from .model import build_model
from .utils import device_from_cfg, get_logger


@torch.no_grad()
def infer(model, loader, device, tta_hflip: bool) -> tuple[list[str], np.ndarray]:
    model.eval()
    ids, probs = [], []
    for x, batch_ids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logit = model(x).squeeze(1)
            if tta_hflip:
                logit = (logit + model(torch.flip(x, dims=[3])).squeeze(1)) / 2
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        ids.extend(batch_ids)
    return ids, np.concatenate(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="experiment run dir with checkpoints/best.pt")
    ap.add_argument("--out", default=None, help="submission path (default <run>/submission.csv)")
    args = ap.parse_args()

    run = Path(args.run)
    # Build the model from the RUN's resolved config (architecture/streams must match the ckpt),
    # falling back to --config for any field the saved one lacks.
    run_cfg_path = run / "config.yaml"
    cfg = load_config(str(run_cfg_path) if run_cfg_path.exists() else args.config)
    log = get_logger("predict")
    device = device_from_cfg(cfg)

    ckpt = torch.load(run / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    log.info("loaded %s (val FREUID %.5f)", run / "checkpoints/best.pt", ckpt["score"]["freuid"])

    test_ids = list_available_test_ids(cfg.paths.data_root)
    log.info("public_test images available: %d", len(test_ids))
    loader = make_test_loader(test_ids, cfg)
    ids, probs = infer(model, loader, device, cfg.predict.tta_hflip)
    pred = dict(zip(ids, probs.astype(float)))

    sub = pd.read_csv(Path(cfg.paths.data_root) / "sample_submission.csv")
    fill = float(cfg.predict.fill_value)
    sub["label"] = sub["id"].map(pred).fillna(fill)
    n_real = sub["id"].isin(pred).sum()
    log.info("submission rows %d | real preds %d | filled %d (=%.2f)",
             len(sub), n_real, len(sub) - n_real, fill)

    out = Path(args.out) if args.out else run / "submission.csv"
    sub.to_csv(out, index=False)
    log.info("wrote %s", out)
    print(out)


if __name__ == "__main__":
    main()
