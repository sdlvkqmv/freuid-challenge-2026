"""timm backbone with single-logit fraud head and optional forensic streams.

Streams (cfg.model.streams), concatenated channel-wise as backbone input:
  rgb  : 3ch  normalized image
  srm  : 3ch  SRM high-pass noise residual (tamper / camera-fingerprint cue)

Rationale (docs/research/forgery_detection.md §2,§6): noise-residual streams add
tamper signal that raw RGB misses; multi-stream (RGB+residual) beats either alone.
"""
from __future__ import annotations

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


class StreamModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        streams = cfg.model.get("streams", ["rgb"])
        self.use_rgb = "rgb" in streams
        self.use_srm = "srm" in streams
        in_ch = (3 if self.use_rgb else 0) + (3 if self.use_srm else 0)
        if in_ch == 0:
            raise ValueError("model.streams must include at least one of: rgb, srm")
        self.srm = SRMResidual() if self.use_srm else None
        self.backbone = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=1,
            in_chans=in_ch,
            drop_rate=cfg.model.get("drop_rate", 0.0),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_rgb:
            parts.append(x)
        if self.use_srm:
            parts.append(self.srm(x))
        return self.backbone(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])


def build_model(cfg) -> nn.Module:
    return StreamModel(cfg)
