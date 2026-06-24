"""Score-level ensemble of multiple runs' submissions.

  python -m freuid.ensemble --out experiments/ens.csv \
      experiments/runA/submission.csv experiments/runB/submission.csv

Rank-averages the fraud scores across submissions. Rank fusion is scale-free and
directly targets the ranking that AuDET / APCER@1%BPCER reward (docs §6 ensembling).
"""
from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subs", nargs="+", help="submission.csv files to fuse")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.subs)
    if len(weights) != len(args.subs):
        raise ValueError("weights count must match submissions count")

    base = pd.read_csv(args.subs[0])[["id"]].copy()
    acc = pd.Series(0.0, index=base.index)
    for w, path in zip(weights, args.subs):
        df = pd.read_csv(path).set_index("id").reindex(base["id"]).reset_index()
        r = df["label"].rank(method="average", pct=True)  # rank-normalize to [0,1]
        acc = acc + w * r.values
    base["label"] = acc / sum(weights)
    base.to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
