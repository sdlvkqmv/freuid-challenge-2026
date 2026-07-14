# Attempts 30–31 — MIL field-crop forensics (CHAMPION)

Status: **done · attempt31-v2 LB 0.03905 🥇** (prev champion 06 = 0.15185; ~3.9× better)

## Motivation
12 consecutive directions failed to beat 06 (a whole-image effb3+SRM+recapture
detector) because they reshaped the *same* whole-image digital-domain model. The
one untried lever with a different signal *geometry* was spatial: the forgery lives
in a **small field region** (photo, a date, a doc-number band), and whole-image
pooling dilutes it. Earlier multicrop attempts (15/16) used a blind grid over the
*whole* image and regressed — they discarded the global recapture cue and cropped
mostly background. The fix: crop **only the informative fields**, per document type,
and aggregate with **max** (one tampered field ⇒ high doc score).

## Pipeline
1. **Per-type field boxes** (`derive_boxes.py` → `session5_artifacts/field_boxes_v2.json`).
   Align each type to a canonical (W,H); build a per-pixel **template mean + std**
   over aligned images; take the top-20%-variance pixels (the ink that changes
   doc-to-doc = the fields), dilate horizontally / close vertically into field
   *bands*, label connected components, keep the **photo** (largest square-ish blob)
   + up to 9 field bands, pad 3%. Yields K≤13 boxes/type. Canonical sizes:
   BENIN/DL (1000,1585), EGYPT/DL (875,1387), GUINEA/DL (1000,1584),
   MAURITIUS/ID (1000,1585), MOZAMBIQUE/DL (630,1000).
2. **MIL scorer** (`stage2_mil_v2.py`). Each doc → K crops resized to 320px. A
   **shared** effb3+SRM scorer (06's backbone+stream) produces a per-crop fraud
   logit; **doc logit = max over the K crops** (multiple-instance, tampered-field
   selects). hflip augmentation in training; **hflip-TTA** (2-view avg) at val and
   inference. +3 static full-doc zone crops appended to the K field crops. 6 epochs,
   LR 3e-4, WD 0.05, pos_weight 1.363, best-val checkpoint selected.

## Results
| # | variant | in-domain val FREUID | LB |
|---|---|---|---|
| 30 | MIL field-crop v1 | ~0 | **0.08023** |
| 31 | v2 320px + hflip aug/TTA + 3 static crops, 6ep | 0.00000 @ep4 | **0.03905 🥇** |

Trajectory: 06 0.15185 → a30 0.08023 → **a31-v2 0.03905**. Rank ~#70/235.

## Why it works (vs the 12 failures)
- **Signal geometry, not signal family.** Same effb3+SRM forensic backbone as 06 —
  but applied to the *field crops* where manipulation concentrates, with max-agg so
  a single edited field drives the score. This is the first change that altered
  *where* the model looks rather than *what* streams/losses/backbone it uses.
- **Template alignment beats blind cropping.** 15/16's whole-image grid failed;
  per-type variance-derived field boxes crop signal, not background.
- **Native-res field detail.** Cropping from the native image (not a downscaled
  whole-doc) preserves the high-frequency forensic cues (edges, JPEG grid, SRM
  residual) inside each field at 320px.

## Follow-ups (session 6, 2026-07-12)
Reseed + capacity ensemble to hedge the known effb3 seed-variance (06 was a lucky
seed): seed123, seed456 (320px b3), 384px b3 seed42, effb4@384 seed42. Rank-avg
ensembles over the 7,821 public ids. See scoreboard for landed LB values.

## Negative side-probes (closed, session 6)
- **Probe A — donor-face reuse** (dhash NN over face crops): negative, closed.
- **Probe B — OCR + field logic rules** (easyocr over 1,500 balanced docs, native
  field crops): **negative.** Doc-number formats are exactly preserved in fraud
  (EGYPT=14 / GUINEA=11 / MAURITIUS=14 digits, 0% deviation), no impossible-date
  signal, no validity-span cluster separation. Forgeries are **visual/pixel-level,
  not semantic-field-logic** — consistent with a field-forensics visual detector
  being the right tool. See scratch/probeB_analyze2.py.
