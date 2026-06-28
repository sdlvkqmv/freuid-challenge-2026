"""Direction D (field-consistency) — photo-identity consistency feature.

Observed dominant fraud family on this dataset = **photo substitution**: the main portrait
is replaced by a different person while the secondary "ghost" portrait (and stated gender)
stay unchanged → the two on-card faces no longer match. This is a SEMANTIC inconsistency
(identity mismatch) that is orthogonal to the pixel-forensic SRM stream and, crucially,
**survives the recapture domain gap** (both faces are recaptured together; their identity
mismatch persists). See docs/research/directions.md (D) and attempt 19.

Feature per image: detect faces (MTCNN), embed each (InceptionResnetV1/vggface2), take the
two largest faces and compute cosine similarity. `face_incons = 1 - sim` (high = likely
swapped). Images with <2 confident faces have NO signal (single-photo DL types) → NaN, to be
filled neutrally at fusion time.

  python -m freuid.face_consistency --split train --sample 6000 --gpu 6   # measure AUROC/coverage
  python -m freuid.face_consistency --split test --gpu 6 --out test_face.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .data import img_path, list_available_test_ids, load_train_df
from .utils import get_logger


def build_face_models(device, min_face_size=20):
    from facenet_pytorch import MTCNN, InceptionResnetV1
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face_size, post_process=True)
    emb = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, emb


@torch.no_grad()
def image_incons(path, mtcnn, emb, device, prob_thr=0.90):
    """Return (face_incons, n_faces). face_incons = 1 - cos(main, ghost); NaN if <2 conf faces."""
    img = Image.open(path).convert("RGB")
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return float("nan"), 0
    keep = [i for i, p in enumerate(probs) if p is not None and p >= prob_thr]
    if len(keep) < 2:
        return float("nan"), len(keep)
    areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in keep]
    order = [keep[j] for j in np.argsort(areas)[::-1]]
    crops = mtcnn.extract(img, boxes[order[:2]], None)        # two largest
    if crops is None or len(crops) < 2:
        return float("nan"), len(keep)
    e = F.normalize(emb(crops.to(device)), dim=1)
    sim = float((e[0] * e[1]).sum())
    return 1.0 - sim, len(keep)


def run(ids, split, mtcnn, emb, device, log):
    rows = []
    for i, img_id in enumerate(ids):
        inc, n = image_incons(img_path("data/extracted", split, img_id), mtcnn, emb, device)
        rows.append({"id": img_id, "face_incons": inc, "n_faces": n})
        if (i + 1) % 500 == 0:
            cov = np.mean([np.isfinite(r["face_incons"]) for r in rows])
            log.info("scored %d/%d | coverage(2-face) %.1f%%", i + 1, len(ids), 100 * cov)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--sample", type=int, default=0, help="train: stratified subsample size (0=all)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log = get_logger("face")
    mtcnn, emb = build_face_models(device)

    if args.split == "train":
        df = load_train_df("data/extracted")
        if args.sample:
            strat = df["type"].astype(str) + "|" + df["label"].astype(str)
            df = df.groupby(strat, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, args.sample // strat.nunique())), random_state=42))
        ids = df["id"].tolist()
        feat = run(ids, "train", mtcnn, emb, device, log)
        feat = feat.merge(df[["id", "label", "type"]], on="id")
        # measure separation on the 2-face (covered) subset
        cov = feat[np.isfinite(feat["face_incons"])]
        log.info("=== coverage by type (2-face %%) ===")
        for t, g in feat.groupby("type"):
            log.info("  %-16s %.1f%% (n=%d)", t, 100 * np.isfinite(g["face_incons"]).mean(), len(g))
        if len(cov) > 10 and cov["label"].nunique() == 2:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(cov["label"], cov["face_incons"])
            log.info("=== face_incons AUROC on 2-face subset: %.4f (n=%d, fraud-rate %.2f) ===",
                     auc, len(cov), cov["label"].mean())
            for t, g in cov.groupby("type"):
                if g["label"].nunique() == 2:
                    log.info("  %-16s AUROC %.4f (n=%d)", t, roc_auc_score(g["label"], g["face_incons"]), len(g))
    else:
        ids = list_available_test_ids("data/extracted")
        feat = run(ids, "test", mtcnn, emb, device, log)

    out = args.out or f"{args.split}_face_consistency.csv"
    feat.to_csv(out, index=False)
    log.info("wrote %s", out)
    print(out)


if __name__ == "__main__":
    main()
