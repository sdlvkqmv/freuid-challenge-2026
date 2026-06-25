# Research Directions — net-new ideas (2026-06-25)

Additive to [`../SUMMARY.md`](../SUMMARY.md) "Remaining directions". Does **not** restate what SUMMARY already prioritizes (push SRM×recapture, DCT stream, real recapture probe, ROI crops, DIRE/FIRE). Only ideas absent from that list, plus one correction.

## Correction — calibration (retraction)

Earlier I suggested "calibrate hard / unsupervised test-time calibration." **Wrong for this metric**, and SUMMARY finding #2 is right: AuDET and APCER@1%BPCER are swept over submitted scores, so any **global monotonic** transform (isotonic/Platt/temperature) is a **no-op**. Don't spend time on it.

**But the salvageable nuance:** a **per-group** transform is *not* globally monotonic, so it **can** move the combined DET curve.
- **Per-doc-type score normalization** (z-score / rank-normalize scores within each predicted doc type before merging into one submission). If the 5 types have mis-aligned score distributions, the global 1%-BPCER operating point is dominated by whichever type runs hottest. Re-aligning per-type can improve APCER@1%BPCER even though it can't help any single type's own DET. **Testable on the recap-val proxy; low cost.** Hypothesis, unproven.

## D. Field-consistency / cross-field semantic checks — net-new, type-agnostic

Absent from current plan. A **non-pixel** signal orthogonal to every forensic stream (RGB/SRM/DCT/DIRE), so it should fuse cleanly.

IDs carry redundant, machine-checkable structure that forgeries (especially inpaint/text edits, the dominant digital family here) tend to break:
- **MRZ ↔ printed fields**: MRZ encodes name/DOB/expiry/doc-number with **ISO 7501 / ICAO 9303 check digits**. Parse MRZ (where present — likely MAURITIUS/ID; African DLs may lack it), verify check digits, cross-check against OCR of the visual zone. Mismatch or failed checksum = strong fraud signal, near-zero false-positive.
- **Intra-field font / baseline consistency**: a spliced/inpainted date or name often has subtly different glyph rendering, baseline, or spacing vs the template. Per-field OCR confidence + glyph-consistency score.
- **Cross-field logical consistency**: issue ≤ expiry, age vs DOB, document-number format per country template.

Why it survives the domain gap: it keys on **semantic/structural** breakage, not camera/compression artifacts → unaffected by the recapture domain shift that breaks the pixel models. Cheap (OCR + rules), fully type-agnostic.

**Pilot:** OCR the 5 templates, build per-type field regex + MRZ checksum, score consistency, **score-fuse late** with the SRM×recapture winner. Measure on recap-val + LB.

## E. Tiny-region max-aggregation (architectural, low cost)

Tampered region is a tiny fraction of pixels; global pooling dilutes it. Per-field (or per-patch) score → **max-aggregate** instead of global average. Complements ROI crops already on the plan; cheap to bolt onto the existing backbone.

# Domain-gap attacks (added 2026-06-25)

The whole game is the train→test domain gap (train ~100% digital edits; test = recapture + private unseen types/generators). Only proven lever so far: forensic-stream × recapture-aug **synergy** (SUMMARY finding #1). D/E above add orthogonal signal; F–J below attack the gap *directly* — most planned items (DCT, ROI, DIRE) do not. Generated via negation + adjacent-possible + reformulation.

## F. Frozen foundation features (CLIP / DINOv2 linear-probe or LoRA) — top bet

> From-scratch EffNet-B3 overfits the digital domain and collapses on unseen generators/types, which is exactly the private-LB objective. Freeze a large pretrained vision encoder and train only a light head, because such features generalize to unseen forgery generators where trained-from-scratch CNNs do not.

Precedent: **UnivFD** (Ojha et al., frozen CLIP + linear probe) beats trained CNNs on *unseen* deepfake generators. Cheap: backbone frozen, train head (or LoRA adapters). Highest leverage vs the actual (OOD) objective.
- **Pilot:** frozen CLIP-ViT-L/14 (and DINOv2-L) → global feature → logistic / 1-layer MLP head, train on full train set. Measure recap-val + LB. Then LoRA the top blocks if linear-probe underfits. ~1 day.

## G. Test-time BN / domain adaptation — cheapest, do first

> Covariate shift between digital train and recapture test is the core failure, and the best model never updates its feature statistics for the test domain. Recompute BatchNorm running stats (or entropy-minimize) on the unlabeled test batch, because that re-aligns features to the recapture domain with no retraining.

Unsupervised, no labels, hours. Bolt onto the 06 winner.
- **Pilot:** at inference, switch BN to use test-batch stats (or Tent: entropy-min over BN affine params, 1–10 steps). Apply to attempt-06 checkpoint. Compare LB vs frozen-stat baseline. Risk: low. Half day.

## H. One-class / anomaly on bona-fide — reframe (F2)

> A discriminative fraud-vs-genuine boundary fits the digital-edit family and won't transfer to the private LB's unseen forgery types. Model only the abundant bona-fide distribution (40k) and score deviation, because genuine documents are far more constrained than the open-ended set of forgery families and a deviation score generalizes to forgery types never seen in training.

Hedges the unseen-type private LB. Medium effort.
- **Pilot:** features from F (frozen encoder) → fit one-class (Mahalanobis / kNN-distance / deep-SVDD) on bona-fide only → distance = fraud score. Score-fuse late with 06. Measure recap-val + LB.

## I. Operating-point-aware hard-negative mining

> The metric's second term is APCER@**1%**BPCER, dominated by the worst ~1% of bona-fide (the false-positive tail), and calibration provably cannot move it (finding #2) — only ranking near the operating point can. Mine the hard bona-fide that sit near the threshold and up-weight loss there, because that sharpens exactly the ranking region the metric scores.

Targets the lever the calibration no-op leaves open. Cheap-ish add to any backbone.
- **Pilot:** train run with hard-negative reweighting (focal or top-k-hardest-bona-fide boosting) on the 06 recipe. Measure APCER@1%BPCER specifically on recap-val + LB.

## J. Domain-adversarial (DANN / gradient reversal) on `is_digital` — hedge

> The model keys on digital-domain artifacts that vanish under recapture. Add a gradient-reversal adversary predicting digital-vs-recapture, because forcing the backbone to be recapture-invariant removes the shortcut that breaks OOD.

Riskier — can erase fraud signal with the domain signal. Try only if G/H stall.
- **Pilot:** DANN head on `is_digital` (use recap-aug to create the recapture class since train has only 20 real). Tune adversary weight. Measure recap-val + LB.

## Priority

1. **G** (test-time BN) — cheapest, do first, free insurance on the 06 winner.
2. **F** (frozen foundation features) — biggest OOD upside; only candidate whose precedent is literally "generalizes to unseen generators."
3. **D** (field-consistency) — strongest orthogonal-signal net-new, domain-gap-robust, cheap; fuse late.
4. **I** (op-point hard-neg mining) — metric squeeze once a backbone wins.
5. **H / J** — hedges for the unseen-type private LB; run if F/G plateau.
E (max-agg) + per-type norm remain low-cost experiments to slot between SUMMARY-priority items.
