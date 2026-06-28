"""Late-fuse the face-consistency signal (direction D) into a base submission (06).

The FREUID metric is rank-based (DET sweep), so we work in rank space. Non-covered images
(no 2-face signal) keep the base model's global rank. Covered images (Mauritius-style dual
photo) get blended: (1-w)*base_rank + w*face_percentile, where face_percentile is the
within-covered-set percentile of `1 - cos(main, ghost)` (high = likely photo substitution).
This nudges suspected swaps up and consistent faces down, only where the signal exists.

  python -m freuid.fuse_face --base experiments/attempt06_.../submission.csv \
      --face test_face_consistency.csv --w 0.4 --out fused.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base submission (e.g. 06 submission.csv)")
    ap.add_argument("--face", required=True, help="test face_consistency.csv (id,face_incons,n_faces)")
    ap.add_argument("--w", type=float, default=0.4, help="face weight on covered images")
    ap.add_argument("--fill", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    face = pd.read_csv(args.face)[["id", "face_incons"]]
    # real-pred ids = those the base actually scored (not the constant fill)
    real = base[base["label"] != args.fill].copy()
    real = real.merge(face, on="id", how="left")

    # base global rank in [0,1]
    real["r_base"] = real["label"].rank(method="average") / len(real)
    covered = real["face_incons"].notna()
    n_cov = int(covered.sum())

    real["score"] = real["r_base"]
    if n_cov > 1:
        # within-covered percentile of face inconsistency
        fc = real.loc[covered, "face_incons"]
        pct = fc.rank(method="average") / len(fc)
        real.loc[covered, "score"] = (1 - args.w) * real.loc[covered, "r_base"] + args.w * pct.values

    fused = dict(zip(real["id"], real["score"]))
    out = base.copy()
    out["label"] = out["id"].map(fused).fillna(args.fill)
    Path(args.out).write_text(out.to_csv(index=False))
    print(f"covered {n_cov}/{len(real)} ({100*n_cov/len(real):.1f}%) | w={args.w} | wrote {args.out}")


if __name__ == "__main__":
    main()
