"""Colorization helpers.

Project convention: near is warm, far is cool. Disparity and relative (inverse) depth
grow toward the camera, so they use `turbo`; metric depth grows away from it, so it uses
`turbo_r`. Every call takes an explicit (vmin, vmax): callers decide the normalization
(e.g. one range frozen over a whole video clip), never this module.
"""
import matplotlib
import numpy as np

INVALID_RGB = (0, 0, 0)


def colorize(values, vmin, vmax, cmap="turbo", invalid_mask=None, invalid_rgb=INVALID_RGB):
    """Map a 2-D float array to uint8 RGB with a fixed range. Pixels in invalid_mask get invalid_rgb (black)."""
    lut = (matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    norm = (np.asarray(values, dtype=np.float32) - vmin) / max(vmax - vmin, 1e-12)
    idx = np.clip(np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0) * 255, 0, 255).astype(np.uint8)
    rgb = lut[idx]
    if invalid_mask is not None:
        rgb[invalid_mask] = invalid_rgb
    return rgb


def robust_range(arrays, lo=2.0, hi=98.0, max_samples=2_000_000, seed=0):
    """Percentile range over the finite values of one or many arrays (used to freeze a clip-wide normalization)."""
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    values = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if values.size > max_samples:
        values = np.random.default_rng(seed).choice(values, max_samples, replace=False)
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))
