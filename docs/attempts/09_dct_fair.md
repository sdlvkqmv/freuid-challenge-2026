# Attempt 09 — DCT stream, FAIR test (06's moderate recapture aug)

Back to [[SUMMARY]] · confounded original [[attempts/08_dct_stream]] · baseline [[attempts/06_effb3_srm_recap]]

## Hypothesis
08 (rgb+srm+dct) regressed to 0.24078 but used the strong/wide aug that 07a proved harmful on its
own → DCT was confounded. Re-run with **06's exact moderate aug** (prob 0.7, narrow ranges,
`configs/dct_recap_fair.yaml`) to isolate the DCT stream's true effect.

## Setup
`tf_efficientnet_b3`, streams=[rgb, srm, dct] (9ch), 384px bs24 8ep, GPU2. recapture = attempt06
settings. Selection by recap_freuid.

## Result
| metric | value |
|---|---|
| in-domain val FREUID | 0.00011 (best ep5) |
| recap-val | 0.00021 |
| **Kaggle public LB** | **0.21476** |

## Takeaway — DCT stream genuinely HURTS (not just confounded)
With the harmful aug removed, DCT still lands 0.21476 vs the same-aug **srm-only** winner 06 (0.15185).
So **adding the DCT block-frequency stream degrades the SRM+recapture model** — confirmed clean
negative. Likely causes: 9ch input dilutes the pretrained 3ch backbone weights further; DCT
band-energy overfits *digital-domain* JPEG-grid artifacts that don't survive the recapture pipeline
(the very domain we need to generalize to); SRM already captures the useful high-frequency cue.
Note recap-val 0.00021 looked *best of all runs* yet LB was near-worst — proxy lied again ([[eval_harness]]).

**Decision:** drop the DCT stream. SRM is the forensic stream that works here. Next lever = a
different signal *family* (field-consistency / MRZ checks, ROI crops), not more pixel streams.
