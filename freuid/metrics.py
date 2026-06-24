"""FREUID Challenge metric.

Positive class = fraud (label 1). Score = fraud probability (higher = more fraud).

Definitions (decision: flag fraud if score >= threshold):
  APCER(t) = P(score <  t | label==1)   # attacks missed   = FNR of positive
  BPCER(t) = P(score >= t | label==0)   # bona-fide flagged = FPR of positive

DET curve = APCER vs BPCER. Both in [0, 1], lower is better.
  AuDET            = area under APCER-vs-BPCER curve (integrate APCER over BPCER)
  APCER@1%BPCER    = APCER interpolated at BPCER = 0.01

FREUID score (lower better):
  g_audet = 1 - AuDET ; g_apcer = 1 - APCER@1%BPCER
  FREUID  = 1 - harmonic_mean(g_audet, g_apcer)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def _det_curve(labels: np.ndarray, scores: np.ndarray):
    # roc_curve: fpr/tpr for the positive class (label 1 = fraud).
    fpr, tpr, thr = roc_curve(labels, scores)
    bpcer = fpr            # bona-fide (neg) flagged as fraud
    apcer = 1.0 - tpr      # frauds (pos) missed
    order = np.argsort(bpcer, kind="stable")
    return apcer[order], bpcer[order], thr[order]


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2 renamed trapz


def audet(apcer: np.ndarray, bpcer: np.ndarray) -> float:
    return float(_trapz(apcer, bpcer))


def apcer_at_bpcer(apcer: np.ndarray, bpcer: np.ndarray, target: float = 0.01) -> float:
    # np.interp needs ascending x (bpcer already sorted ascending)
    return float(np.interp(target, bpcer, apcer))


def freuid_score(labels, scores, bpcer_target: float = 0.01) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    apcer, bpcer, _ = _det_curve(labels, scores)
    au = audet(apcer, bpcer)
    ap = apcer_at_bpcer(apcer, bpcer, bpcer_target)
    g1, g2 = 1.0 - au, 1.0 - ap
    hm = 0.0 if (g1 + g2) == 0 else 2.0 * g1 * g2 / (g1 + g2)
    return {"freuid": 1.0 - hm, "audet": au, "apcer@1%bpcer": ap}
