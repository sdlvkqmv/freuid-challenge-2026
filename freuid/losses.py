"""Operating-point-aware losses for direction I (hard-negative mining).

The FREUID metric is driven by APCER@1%BPCER: the threshold is set by the top ~1% of
*bona-fide* scores, and APCER = fraction of *fraud* falling below it. So the samples that
matter are the **hard negatives** (high-scoring bona-fide that raise the threshold) and the
**hard positives** (low-scoring fraud that get missed). Plain BCE spends most gradient on
easy examples; these losses concentrate it on the hard tail near the operating point.

  focal : down-weight easy examples by (1-p_t)^gamma (Lin 2017). gamma_pos/gamma_neg let
          the fraud and bona-fide tails be focused independently (asymmetric, Ben-Baruch 2020).
  ohem  : online hard example mining — backprop only the hardest `ohem_frac` of each batch.

Both reduce to weighted BCE when gamma=0 / ohem_frac=1.0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _focal_per_sample(logit, target, alpha, gamma_pos, gamma_neg):
    """Per-sample focal BCE. logit,target: (B,). alpha = weight on positives."""
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    # p_t = p for positives, (1-p) for negatives; modulating (1-p_t)^gamma
    p_t = target * p + (1 - target) * (1 - p)
    gamma = target * gamma_pos + (1 - target) * gamma_neg
    mod = (1 - p_t).clamp(min=1e-6) ** gamma
    w = target * alpha + (1 - target) * (1 - alpha)
    return w * mod * ce


def build_loss(cfg, pos_weight: float):
    """Return a callable(logit, target) -> scalar loss from cfg.train.loss.

    Backward compatible: no cfg.train.loss (or type=bce, gamma=0) -> BCE with pos_weight.
    """
    spec = cfg.train.get("loss") or {}
    ltype = spec.get("type", "bce")
    ohem_frac = float(spec.get("ohem_frac", 1.0))

    if ltype == "focal":
        gamma_pos = float(spec.get("gamma_pos", spec.get("gamma", 2.0)))
        gamma_neg = float(spec.get("gamma_neg", spec.get("gamma", 2.0)))
        # alpha: balance positives; default derives from pos_weight so it matches BCE baseline
        alpha = spec.get("alpha")
        alpha = pos_weight / (1.0 + pos_weight) if alpha is None else float(alpha)

        def loss_fn(logit, target):
            per = _focal_per_sample(logit, target, alpha, gamma_pos, gamma_neg)
            return _reduce(per, ohem_frac)
    else:  # bce + optional ohem
        pw = torch.tensor([float(pos_weight)])

        def loss_fn(logit, target):
            per = F.binary_cross_entropy_with_logits(
                logit, target, pos_weight=pw.to(logit.device), reduction="none")
            return _reduce(per, ohem_frac)

    return loss_fn


def _reduce(per_sample, ohem_frac: float):
    if ohem_frac >= 1.0:
        return per_sample.mean()
    k = max(1, int(per_sample.numel() * ohem_frac))
    hard, _ = torch.topk(per_sample, k)          # hardest k by loss
    return hard.mean()
