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
        # --- enhanced-realism cues (gated, default off; backward-compatible with 06) ---
        self.chroma_moire = c.get("chroma_moire", False)        # per-channel multiplicative moiré
        self.chroma_ab = c.get("chroma_ab", 0.0)                # chromatic aberration, max px radial shift
        self.vignette = c.get("vignette", 0.0)                  # max radial darkening strength

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

        # 3. moiré beat pattern (camera sensor grid vs print/display raster)
        if r.rand() < self.moire_p:
            x = _moire_chroma(x, self.moire_amp, r) if self.chroma_moire else _moire(x, self.moire_amp, r)

        # 3b. chromatic aberration (lens dispersion: R/B fringe radially at edges)
        if self.chroma_ab > 0:
            x = _chroma_ab(x, int(round(r.uniform(0, self.chroma_ab))))

        # 4. illumination / gamma / white-balance (printer + camera response)
        g = r.uniform(*self.gamma)
        x = 255.0 * np.power(np.clip(x / 255.0, 0, 1), g)
        x *= r.uniform(*self.bright)
        x *= (1.0 + r.uniform(-self.wb, self.wb, size=3))[None, None, :]

        # 4b. vignetting (lens light falloff toward corners)
        if self.vignette > 0:
            x = _vignette(x, r.uniform(0, self.vignette))

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


def _moire_chroma(x: np.ndarray, amp: float, r) -> np.ndarray:
    """Per-channel MULTIPLICATIVE moiré — models the colored beat between the display
    RGB subpixel layout and the camera sensor grid (a hallmark recapture artifact). Two
    superimposed frequencies (display pitch + sensor pitch) with a per-channel phase shift."""
    h, w = x.shape[:2]
    f1, f2 = r.uniform(0.15, 0.5), r.uniform(0.15, 0.5)
    t1, t2 = r.uniform(0, np.pi), r.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    base1 = 2 * np.pi * f1 * (xx * np.cos(t1) + yy * np.sin(t1))
    base2 = 2 * np.pi * f2 * (xx * np.cos(t2) + yy * np.sin(t2))
    out = x.copy()
    m = amp / 255.0                              # modulation depth
    for c in range(3):
        ph = r.uniform(0, 2 * np.pi)             # RGB subpixel offset
        grid = 0.5 * (np.sin(base1 + ph) + np.sin(base2 + ph))
        out[..., c] = x[..., c] * (1.0 + m * grid)
    return out


def _chroma_ab(x: np.ndarray, d: int) -> np.ndarray:
    """Chromatic aberration: shift R/B channels oppositely (lens dispersion fringe)."""
    if d <= 0:
        return x
    x = x.copy()
    x[..., 0] = np.roll(x[..., 0], d, axis=1)
    x[..., 2] = np.roll(x[..., 2], -d, axis=1)
    return x


def _vignette(x: np.ndarray, strength: float) -> np.ndarray:
    """Radial light falloff toward the corners (camera lens vignetting)."""
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2.0)
    factor = (1.0 - strength * rad ** 2)[..., None]
    return x * factor
