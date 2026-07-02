"""Direction T1 — VLM zero-shot forgery score (a genuinely different paradigm).

Every model so far (06 included) LEARNS the fraud signal from the 69k train set, which is
99.97% digital → any learned signal is digital-domain and collapses on the recapture-heavy
test (root finding #0). This bypasses training entirely: an open VLM (Qwen2-VL-2B), pretrained
on web-scale REAL photographs, judges each document. Two consequences we want:
  (a) recapture/print-and-capture is IN-distribution for the VLM (unlike our train set) →
      domain-gap-robust by construction;
  (b) it is a completely different function from a pixel-forensic CNN → error-decorrelated,
      so rank-fusion with 06 can actually help (all 10 prior bolt-ons were >0.9 corr → useless).

Score per image = P("Yes") vs P("No") on the first generated token of a forensic yes/no
question, read from the logits (continuous, no sampling). No fraud labels are used.

  # 4-way shard across free GPUs, then concat:
  for s in 0 1 2 3; do python -m freuid.vlm_score --split test --gpu $((s+3)) \
      --nshard 4 --shard $s --out vlm_test_$s.csv & done; wait
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import img_path, list_available_test_ids, load_train_df
from .utils import get_logger

PROMPT = (
    "You are a forensic document examiner. The image is a photograph of an identity document "
    "(an ID card or driver's license). Inspect it for any sign of forgery or manipulation: a "
    "portrait that has been replaced, pasted, or does not match the other photo/fields; "
    "misaligned, inconsistent, or re-typed text; altered dates or numbers; splicing, cloning, "
    "or blending artifacts; irregular fonts, spacing, or background security patterns. "
    "Is this identity document forged or manipulated in any way? Answer with one word: Yes or No."
)


def build_vlm(device, model_id="Qwen/Qwen2-VL-2B-Instruct", max_pixels=768 * 768):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="eager"
    ).eval().to(device)
    # cap vision tokens for speed (ID crops are large); min keeps small text legible
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 256, max_pixels=max_pixels)
    return model, processor


def _yes_no_ids(processor):
    """Token ids whose decoded form starts a Yes / No answer (handle leading-space variants)."""
    tok = processor.tokenizer
    yes, no = set(), set()
    for w in ["Yes", " Yes", "yes", " yes", "YES"]:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            yes.add(ids[0])
    for w in ["No", " No", "no", " no", "NO"]:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            no.add(ids[0])
    return sorted(yes), sorted(no)


@torch.no_grad()
def score_image(path, model, processor, yes_ids, no_ids, device):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{Path(path).resolve()}"},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1, :].float()          # next-token distribution
    yes_l = torch.logsumexp(logits[yes_ids], 0)
    no_l = torch.logsumexp(logits[no_ids], 0)
    # P(Yes) over the {Yes,No} restricted head → calibrated-ish fraud probability
    return float(torch.softmax(torch.stack([no_l, yes_l]), 0)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--sample", type=int, default=0, help="train: stratified subsample for AUROC")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log = get_logger(f"vlm{args.shard}")
    model, processor = build_vlm(device)
    yes_ids, no_ids = _yes_no_ids(processor)
    log.info("yes_ids=%s no_ids=%s", yes_ids, no_ids)

    if args.split == "train":
        df = load_train_df("data/extracted")
        if args.sample:
            strat = df["type"].astype(str) + "|" + df["label"].astype(str)
            df = df.groupby(strat, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, args.sample // strat.nunique())), random_state=42))
        ids = df["id"].tolist()
        labels = dict(zip(df["id"], df["label"]))
        types = dict(zip(df["id"], df["type"]))
    else:
        ids = list_available_test_ids("data/extracted")
        labels = types = None

    ids = [x for i, x in enumerate(ids) if i % args.nshard == args.shard]
    rows = []
    for i, img_id in enumerate(ids):
        try:
            s = score_image(img_path("data/extracted", args.split, img_id), model, processor,
                            yes_ids, no_ids, device)
        except Exception as e:  # noqa: BLE001 — keep going; missing score filled neutrally at fuse
            log.warning("id %s failed: %s", img_id, e)
            s = float("nan")
        row = {"id": img_id, "vlm_fraud": s}
        if labels is not None:
            row["label"] = labels[img_id]
            row["type"] = types[img_id]
        rows.append(row)
        if (i + 1) % 200 == 0:
            log.info("scored %d/%d (shard %d)", i + 1, len(ids), args.shard)

    feat = pd.DataFrame(rows)
    if args.split == "train" and feat["label"].nunique() == 2:
        from sklearn.metrics import roc_auc_score
        good = feat[feat["vlm_fraud"].notna()]
        log.info("=== VLM AUROC (all): %.4f (n=%d) ===",
                 roc_auc_score(good["label"], good["vlm_fraud"]), len(good))
        for t, g in good.groupby("type"):
            if g["label"].nunique() == 2:
                log.info("  %-16s AUROC %.4f (n=%d)", t, roc_auc_score(g["label"], g["vlm_fraud"]), len(g))
    feat.to_csv(args.out, index=False)
    log.info("wrote %s (%d rows)", args.out, len(feat))
    print(args.out)


if __name__ == "__main__":
    main()
