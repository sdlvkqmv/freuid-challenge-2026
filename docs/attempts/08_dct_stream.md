# Attempt 08 — add a DCT block-frequency stream (rgb+srm+dct)

Back to [[SUMMARY]] · proxy [[eval_harness]] · fair re-test [[attempts/09_dct_fair]]

## Hypothesis
Double-JPEG / block periodicity is a core print-and-capture signature (research §2, CAT-Net),
complementary to SRM's pixel-noise residual. Add a `dct` stream = local 8×8 block-DCT band energy
(low/mid/high AC), 3ch, computed by `DCTResidual` in `freuid/model.py`. streams=[rgb, srm, dct] (9ch).

## Setup
`configs/srm_recap_strong.yaml`, streams=[rgb,srm,dct], 384px bs24 12ep, GPU5.
⚠️ Used the **strong/wide recapture aug** (prob 0.85) — same config 07a proved harmful.

## Result
| metric | value |
|---|---|
| recap-val | 0.00063 |
| **Kaggle public LB** | **0.24078** | 🔴 worst of the session |

## Takeaway — CONFOUNDED
0.24078 is bad, but **the regression is confounded**: 08 inherited the strong/wide aug that 07a
showed wrecks even the plain SRM model (0.15185→0.19546). So this does **not** prove DCT is bad —
it bundles (DCT stream) + (harmful aug). → [[attempts/09_dct_fair]] re-tests DCT with 06's moderate
aug to isolate the stream's true effect. The DCT stream implementation itself is verified
(forward/shape/finite checks passed).
