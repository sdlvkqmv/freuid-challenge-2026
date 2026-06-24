"""Print-and-capture (recapture) simulation.

Train is ~all digital (69,332 digital / 20 recaptured) but the FREUID test emphasizes
print-and-capture images (research §4). This transform synthesizes the recapture pipeline on
digital training images so the model learns fraud cues that survive printing + recapture,
instead of digital-domain shortcuts (see docs finding #0).

Pipeline modeled (each applied stochastically): printer/display halftone-ish resample ->
optics defocus -> moiré beat pattern -> illumination/gamma/white-balance shift -> sensor noise
-> JPEG recompression. Operates on a PIL.Image, returns a PIL.Image. Insert AFTER geometric
crops and BEFORE ToTensor/Normalize.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _rng(seed_src) -> np.random.RandomState:
    # per-call randomness without global Date/Random; vary by pixel hash so workers differ
    return np.random.RandomState(seed_src & 0x7FFFFFFF)


class RecaptureSim:
    def __init__(self, cfg):
        c = cfg
        self.p = c.get("prob", 0.5)
        self.downscale = tuple(c.get("downscale", [0.5, 1.0]))   # min/max relative size
        self.blur = tuple(c.get("blur", [0.0, 1.2]))             # gaussian sigma range
        self.moire_p = c.get("moire_p", 0.3)
        self.moire_amp = c.get("moire_amp", 12.0)
        self.gamma = tuple(c.get("gamma", [0.8, 1.2]))
        self.bright = tuple(c.get("bright", [0.85, 1.15]))
        self.wb = c.get("wb", 0.06)                              # per-channel white-balance jitter
        self.noise_std = tuple(c.get("noise_std", [0.0, 8.0]))  # sensor noise (0-255 scale)
        self.jpeg_q = tuple(c.get("jpeg_q", [40, 92]))

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img)
        r = _rng(int(a[:4, :4, 0].sum()) * 2654435761 + a.size)
        if r.rand() > self.p:
            return img
        w, h = img.size

        # 1. resample loss (print raster + camera downsample then back up)
        s = r.uniform(*self.downscale)
        if s < 0.99:
            small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)

        # 2. optics defocus (PIL C-optimized)
        sig = r.uniform(*self.blur)
        if sig > 0.05:
            img = img.filter(ImageFilter.GaussianBlur(radius=sig))

        x = np.asarray(img).astype(np.float32)

        # 3. moiré beat pattern (camera sensor grid vs print raster)
        if r.rand() < self.moire_p:
            x = _moire(x, self.moire_amp, r)

        # 4. illumination / gamma / white-balance (printer + camera response)
        g = r.uniform(*self.gamma)
        x = 255.0 * np.power(np.clip(x / 255.0, 0, 1), g)
        x *= r.uniform(*self.bright)
        x *= (1.0 + r.uniform(-self.wb, self.wb, size=3))[None, None, :]

        # 5. sensor noise
        ns = r.uniform(*self.noise_std)
        if ns > 0.1:
            x = x + r.randn(*x.shape) * ns

        out = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))

        # 6. JPEG recompression (capture + storage)
        q = int(r.randint(self.jpeg_q[0], self.jpeg_q[1] + 1))
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def _moire(x: np.ndarray, amp: float, r) -> np.ndarray:
    h, w = x.shape[:2]
    f = r.uniform(0.15, 0.5)                     # spatial frequency (cycles/pixel)
    theta = r.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.sin(2 * np.pi * f * (xx * np.cos(theta) + yy * np.sin(theta)))
    return x + amp * grid[..., None]
