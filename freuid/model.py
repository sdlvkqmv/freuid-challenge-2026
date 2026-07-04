"""timm backbone with single-logit fraud head and optional forensic streams.

Streams (cfg.model.streams), concatenated channel-wise as backbone input:
  rgb  : 3ch  normalized image
  srm  : 3ch  SRM high-pass noise residual (tamper / camera-fingerprint cue)
  dct  : 3ch  local 8x8 block-DCT band energy (low/mid/high AC) — JPEG/double-JPEG
              periodicity, a core print-and-capture (recapture) signature (research §2, CAT-Net)

Rationale (docs/research/forgery_detection.md §2,§6): noise-residual streams add
tamper signal that raw RGB misses; multi-stream (RGB+residual) beats either alone.
The DCT stream is complementary to SRM: SRM captures pixel-domain noise, DCT captures
block-frequency artifacts that double-compression (print->display->recapture) imprints.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3 canonical SRM high-pass kernels (Fridrich/Kodovsky), 5x5, normalized.
_SRM_KERNELS = [
    # 1st-order edge
    [[0, 0, 0, 0, 0],
     [0, -1, 2, -1, 0],
     [0, 2, -4, 2, 0],
     [0, -1, 2, -1, 0],
     [0, 0, 0, 0, 0]],
    # 2nd-order
    [[-1, 2, -2, 2, -1],
     [2, -6, 8, -6, 2],
     [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    # 3x3 high-pass (zero-padded to 5x5)
    [[0, 0, 0, 0, 0],
     [0, 1, -2, 1, 0],
     [0, -2, 4, -2, 0],
     [0, 1, -2, 1, 0],
     [0, 0, 0, 0, 0]],
]
_SRM_NORM = [4.0, 12.0, 4.0]


class SRMResidual(nn.Module):
    """Fixed high-pass conv producing a 3-channel noise residual from RGB."""

    def __init__(self):
        super().__init__()
        w = torch.zeros(3, 3, 5, 5)
        for i, (k, n) in enumerate(zip(_SRM_KERNELS, _SRM_NORM)):
            kt = torch.tensor(k, dtype=torch.float32) / n
            for c in range(3):          # same filter applied per RGB channel
                w[i, c] = kt / 3.0
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=2)


def _dct_basis_2d() -> torch.Tensor:
    """64 separable 8x8 DCT-II basis filters as a (64,1,8,8) conv weight."""
    n = torch.arange(8, dtype=torch.float32)
    k = n.view(8, 1)
    b1d = torch.cos(math.pi * (2 * n + 1) * k / 16.0)          # (8 freqs, 8 taps)
    b1d *= math.sqrt(2.0 / 8.0)
    b1d[0] *= math.sqrt(1.0 / 2.0)                              # k=0 normalization
    w = torch.zeros(64, 1, 8, 8)
    for u in range(8):
        for v in range(8):
            w[u * 8 + v, 0] = torch.outer(b1d[u], b1d[v])
    return w


class DCTResidual(nn.Module):
    """Local 8x8 block-DCT band-energy map (3ch): low/mid/high AC frequency energy.

    DC and very-low components are dropped (they track local brightness, not artifacts).
    Captures JPEG-grid / double-compression periodicity that survives recapture.
    """

    _LUMA = (0.299, 0.587, 0.114)

    def __init__(self):
        super().__init__()
        self.register_buffer("dct", _dct_basis_2d())           # (64,1,8,8)
        self.register_buffer("luma", torch.tensor(self._LUMA).view(1, 3, 1, 1))
        # band assignment by zig-zag-ish radius (u+v); index 0 (DC) excluded.
        low, mid, high = [], [], []
        for u in range(8):
            for v in range(8):
                idx = u * 8 + v
                if idx == 0:
                    continue
                r = u + v
                (low if r <= 2 else mid if r <= 6 else high).append(idx)
        self.register_buffer("low_idx", torch.tensor(low))
        self.register_buffer("mid_idx", torch.tensor(mid))
        self.register_buffer("high_idx", torch.tensor(high))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = (x * self.luma).sum(1, keepdim=True)            # (B,1,H,W)
        coef = F.conv2d(gray, self.dct, padding=4)             # (B,64,H+1,W+1)
        coef = coef[..., : x.shape[-2], : x.shape[-1]]
        e = coef * coef
        bands = [e.index_select(1, idx).mean(1, keepdim=True)
                 for idx in (self.low_idx, self.mid_idx, self.high_idx)]
        return torch.log1p(torch.cat(bands, dim=1))            # (B,3,H,W), compress dynamic range


class StreamModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        streams = cfg.model.get("streams", ["rgb"])
        self.use_rgb = "rgb" in streams
        self.use_srm = "srm" in streams
        self.use_dct = "dct" in streams
        in_ch = 3 * (self.use_rgb + self.use_srm + self.use_dct)
        if in_ch == 0:
            raise ValueError("model.streams must include at least one of: rgb, srm, dct")
        self.srm = SRMResidual() if self.use_srm else None
        self.dct_stream = DCTResidual() if self.use_dct else None
        self.backbone = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=1,
            in_chans=in_ch,
            drop_rate=cfg.model.get("drop_rate", 0.0),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.0),
        )
        # trade compute for memory: lets large backbones (convnext_large etc.) fit the
        # 10GB GPUs at usable batch size. timm exposes this on every supported backbone.
        if cfg.model.get("grad_checkpointing", False):
            self.backbone.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_rgb:
            parts.append(x)
        if self.use_srm:
            parts.append(self.srm(x))
        if self.use_dct:
            parts.append(self.dct_stream(x))
        return self.backbone(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])


class FrozenFoundation(nn.Module):
    """Frozen foundation backbone (CLIP / DINOv2 ViT) + light trainable head.

    Direction F (docs/research/directions): UnivFD precedent — a frozen CLIP-ViT-L/14
    backbone + linear probe generalizes to *unseen* generators, which is exactly the
    private-LB objective (2 unseen doc types). The pretrained prior is the OOD lever; we
    only learn a cheap head on top. Backbone is permanently in eval mode and runs under
    no_grad (no stochastic depth, no grad memory) regardless of model.train().

    head: 'linear' (UnivFD) or 'mlp' (one hidden layer + GELU + dropout).
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=0,                     # feature extractor (pooled)
            img_size=cfg.data.img_size,        # interpolate pos-embed (DINOv2 default 518)
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        feat = self.backbone.num_features
        head = cfg.model.get("head", "linear")
        drop = cfg.model.get("drop_rate", 0.0)
        if head == "mlp":
            hid = cfg.model.get("head_hidden", 512)
            self.head = nn.Sequential(
                nn.Dropout(drop), nn.Linear(feat, hid), nn.GELU(),
                nn.Dropout(drop), nn.Linear(hid, 1),
            )
        elif head == "linear":
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(feat, 1))
        else:
            raise ValueError(f"unknown model.head: {head}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()                   # keep backbone frozen/deterministic
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)
        return self.head(feat)


def build_model(cfg) -> nn.Module:
    if cfg.model.get("frozen", False):
        return FrozenFoundation(cfg)
    return StreamModel(cfg)
