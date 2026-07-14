#!/usr/bin/env python3
"""
FREUID Challenge 2026 — inference entrypoint (team submission).

Pipeline (matches our final Kaggle submission, attempt31 "MIL field-crop v2"):
  1. For every image in the flat data dir, compute a 1024-bit dhash.
  2. Assign a document type via nearest-neighbour Hamming match against the
     69,352 training-image hashes bundled in ``assets/train_dhash_refs.npz``
     (documents are template-locked per type, so NN type assignment is
     near-perfect).
  3. Resize the image to its type's canonical resolution, cut the 13
     per-type field crops (photo/name/DOB/number/MRZ/static zones) defined in
     ``assets/field_boxes_v2.json``, resize each to 320px.
  4. Score each crop with an EfficientNet-B3 fed RGB+SRM (6ch); the document
     fraud logit is the MAX over crops, with horizontal-flip TTA (2 views).

Sandbox contract:
  /data/          read-only, flat directory of image files ({id}.{ext})
  /submissions/   read-write, must contain submission.csv (id,label) on exit

Local multi-GPU sharding: --shard K --nshards N writes only rows for images
with index % N == K (merge shards afterwards).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FREUID_DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("FREUID_OUTPUT_DIR", "/submissions"))
SUBMISSION_PATH = Path(os.environ.get("FREUID_SUBMISSION_PATH", OUTPUT_DIR / "submission.csv"))

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CROP = 320
KMAX = 13
CANON = {
    "BENIN/DL": (1000, 1585),
    "EGYPT/DL": (875, 1387),
    "GUINEA/DL": (1000, 1584),
    "MAURITIUS/ID": (1000, 1585),
    "MOZAMBIQUE/DL": (630, 1000),
}
IMEAN = (0.485, 0.456, 0.406)
ISTD = (0.229, 0.224, 0.225)

# --- model (must mirror training-time module tree so state_dict keys match) ---

_SRM_KERNELS = [
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, -2, 4, -2, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0]],
]
_SRM_NORM = [4.0, 12.0, 4.0]


class SRMResidual(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.zeros(3, 3, 5, 5)
        for i, (k, n) in enumerate(zip(_SRM_KERNELS, _SRM_NORM)):
            kt = torch.tensor(k, dtype=torch.float32) / n
            for c in range(3):
                w[i, c] = kt / 3.0
        self.register_buffer("weight", w)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2)


class StreamModel(nn.Module):
    """EfficientNet-B3 over channel-concatenated [RGB, SRM residual] (6ch)."""

    def __init__(self):
        super().__init__()
        import timm

        self.srm = SRMResidual()
        self.backbone = timm.create_model(
            "tf_efficientnet_b3", pretrained=False, num_classes=1,
            in_chans=6, drop_rate=0.2, drop_path_rate=0.1,
        )

    def forward(self, x):
        return self.backbone(torch.cat([x, self.srm(x)], dim=1))


class MIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = StreamModel()

    def forward(self, x):  # x (B,K,3,H,W) -> per-crop logits (B,K)
        B, K = x.shape[:2]
        return self.scorer(x.reshape(B * K, *x.shape[2:])).squeeze(1).view(B, K)


# --- type assignment via dhash nearest neighbour ---

def dhash1024(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((33, 32), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).astype(np.uint8).ravel()


def assign_types(paths, refs_npz: Path, device) -> list:
    refs = np.load(refs_npz, allow_pickle=True)
    ref_bits = np.unpackbits(refs["hashes"], axis=1)[:, :1024]
    ref_types = [str(t) for t in refs["types"]]
    B = torch.from_numpy(ref_bits.astype(np.float32)).to(device) * 2 - 1
    out = []
    chunk = 512
    for i in range(0, len(paths), chunk):
        bits = np.stack([dhash1024(p) for p in paths[i : i + chunk]])
        A = torch.from_numpy(bits.astype(np.float32)).to(device) * 2 - 1
        with torch.no_grad():
            ix = (A @ B.T).argmax(dim=1).cpu().numpy()
        out.extend(ref_types[j] for j in ix)
        print(f"  type-assign {min(i + chunk, len(paths))}/{len(paths)}", file=sys.stderr)
    return out


# --- crop dataset ---

class CropSet(Dataset):
    def __init__(self, rows, boxes):
        self.rows = rows  # (id, path, type)
        self.boxes = boxes
        self.tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMEAN, ISTD)]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rid, path, typ = self.rows[i]
        im = Image.open(path).convert("RGB")
        W, H = CANON[typ]
        if im.size != (W, H):
            im = im.resize((W, H))
        cs = [im.crop((x0, y0, x1, y1)).resize((CROP, CROP)) for y0, y1, x0, x1 in self.boxes[typ]]
        while len(cs) < KMAX:
            cs.append(cs[-1])
        return torch.stack([self.tfm(c) for c in cs[:KMAX]]), rid


def parse_args():
    p = argparse.ArgumentParser(description="Generate FREUID submission.csv")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    p.add_argument("--weights", type=Path, default=APP_DIR / "assets" / "best.pt")
    p.add_argument("--refs", type=Path, default=APP_DIR / "assets" / "train_dhash_refs.npz")
    p.add_argument("--boxes", type=Path, default=APP_DIR / "assets" / "field_boxes_v2.json")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def discover_images(data_dir: Path):
    pairs = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            pairs.append((path.stem, path))
    if not pairs:
        raise FileNotFoundError(f"No images found in {data_dir}")
    return pairs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", file=sys.stderr)

    pairs = discover_images(args.data_dir.resolve())
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    print(f"{len(pairs)} images (shard {args.shard}/{args.nshards})", file=sys.stderr)

    types = assign_types([p for _, p in pairs], args.refs, device)
    rows = [(rid, path, typ) for (rid, path), typ in zip(pairs, types)]

    boxes = json.loads(args.boxes.read_text())
    model = MIL().to(device)
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    loader = DataLoader(
        CropSet(rows, boxes), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    ids, scores = [], []
    with torch.no_grad():
        for n, (x, bid) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                logits = 0.5 * (logits + model(torch.flip(x, dims=[-1])))  # hflip TTA
                doc = logits.max(dim=1).values
            scores.append(torch.sigmoid(doc).float().cpu().numpy())
            ids.extend(bid)
            if (n + 1) % 50 == 0:
                print(f"  scored {len(ids)}/{len(rows)}", file=sys.stderr)

    out = pd.DataFrame({"id": ids, "label": np.concatenate(scores).astype(float)})
    if not np.isfinite(out["label"].to_numpy()).all():
        raise ValueError("Non-finite labels produced.")
    if set(out["id"]) != {rid for rid, _, _ in rows}:
        raise ValueError("id mismatch between discovered images and scored rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
