"""Real-recapture probe — the only NON-simulated recapture signal in train.

Train has just 20 rows with is_digital=False (real print-and-capture): 14 fraud / 6 bona-fide.
That is far too few for FREUID's APCER@1%BPCER (needs ~>=100 negatives), so a real-recapture
*FREUID* proxy is not statistically meaningful. What IS meaningful at n=20 is a rank-separation
check: does the model rank these real recaptured fraud images above the real bona-fide ones?

Reports, on the 20 real-recapture rows (clean transform, no sim aug):
  - AUROC (14x6 = 84 pairs; rank-based, the most stable thing at this n)
  - mean fraud score vs mean bona-fide score (+ gap)
This is a directional sanity check (is the recapture domain handled at all?), NOT a selection
metric. Use it to flag models that collapse on the real recapture domain. See docs finding #0.

  python -m freuid.probe --run experiments/<run_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import load_config
from .data import FraudDataset, build_transforms, load_train_df
from .model import build_model
from .utils import device_from_cfg, get_logger


@torch.no_grad()
def _scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(str(run / "config.yaml"))
    log = get_logger("probe")
    device = device_from_cfg(cfg)

    df = load_train_df(cfg.paths.data_root)
    real = df[~df["is_digital"]].reset_index(drop=True)
    log.info("real-recapture rows: %d (fraud %d / bona-fide %d)",
             len(real), int((real.label == 1).sum()), int((real.label == 0).sum()))

    ds = FraudDataset(real, cfg.paths.data_root, "train",
                      build_transforms(cfg, train=False), has_label=True)
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, pin_memory=True)

    ckpt = torch.load(run / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    probs, labels = _scores(model, loader, device)
    pos, neg = probs[labels == 1], probs[labels == 0]
    auroc = roc_auc_score(labels, probs) if len(neg) and len(pos) else float("nan")
    log.info("AUROC(real recap) %.4f | mean fraud %.4f vs bona-fide %.4f | gap %.4f",
             auroc, pos.mean(), neg.mean(), pos.mean() - neg.mean())
    print(f"{run.name}\tAUROC={auroc:.4f}\tgap={pos.mean() - neg.mean():.4f}")


if __name__ == "__main__":
    main()
