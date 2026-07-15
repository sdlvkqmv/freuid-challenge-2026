# Training provenance

Archived verbatim from the two final training run directories to document
that both selected models were trained on **2026-07-08**, before the
July 13 private-test release / code freeze:

- `attempt31_v2/` — primary model (`assets/best.pt`): run
  `attempt31_mil_fieldcrop_v2_320px_6ep_hflip_static_20260708_221657`,
  training log timestamps 2026-07-08/09, best-val checkpoint epoch 4.
- `attempt30_v1/` — second selection (`assets/best_a30.pt`): run
  `attempt30_mil_fieldcrop_effb3srm_20260708_203539`, trained 2026-07-08.

The corresponding Kaggle submissions predate the freeze as well
(public rows: 54470992 / 2026-07-08 and the attempt30 submission /
2026-07-08). Post-release work was limited to inference on the private
images and packaging: the two final submissions (54691560, 54710765)
have public-test rows bit-identical to the pre-freeze submissions, which
is only possible with unchanged weights.

Checkpoint SHA-256 hashes: `assets/CHECKSUMS.txt`.
