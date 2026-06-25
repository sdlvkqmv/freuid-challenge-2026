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

## Priority

Field-consistency (D) is the strongest net-new bet — orthogonal signal, domain-gap-robust, cheap. Per-type normalization and max-agg are low-cost experiments to slot between the SUMMARY-priority items.
