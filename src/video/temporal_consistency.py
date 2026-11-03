"""Optional numeric temporal check for Part 2.2 (the handout lists it as optional).

For each consecutive pair (t, t+1): RAFT-large flow t->t+1 and t+1->t on the RGB frames,
forward-backward check to drop occluded / unreliable pixels, then backward-warp the depth of
frame t+1 onto frame t and measure |d_t - warp(d_{t+1})|. Depth is first mapped to [0, 1] with
each method's frozen clip-wide normalization (same rule as make_video_figures.py), so the error
is a fraction of that method's range. Both methods are scored on the same pixels; sky (where
the frame-wise model outputs exactly 0) is excluded.

No plotting here: on Windows, matplotlib text layout after torch has loaded its OpenMP runtime
aborts with a duplicate libiomp5md.dll. make_video_figures.py draws the per-pair curve.

Usage: python -I src/video/temporal_consistency.py   (run before make_video_figures.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.utils import ROOT
from src.utils.viz import robust_range

OUT = ROOT / "outputs/video"
METHODS = [("dav2_small_framewise", "Depth Anything V2-S (frame-wise)"), ("vda_small", "Video Depth Anything-S")]
FB_TOL = (0.01, 0.5)  # |f + b|^2 < 0.01 (|f|^2 + |b|^2) + 0.5, the usual occlusion test


def warp(x, flow):
    """Sample x (B,C,H,W) at p + flow(p)."""
    _, _, h, w = x.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    gx = (xs + flow[:, 0]) / (w - 1) * 2 - 1
    gy = (ys + flow[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(x, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="zeros", align_corners=True)


@torch.no_grad()
def flows(frames):
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).cuda().eval()
    tf = weights.transforms()
    fwd, bwd = [], []
    for t in range(len(frames) - 1):
        a = torch.from_numpy(frames[t]).permute(2, 0, 1)[None]
        b = torch.from_numpy(frames[t + 1]).permute(2, 0, 1)[None]
        a, b = tf(a, b)
        fwd.append(model(a.cuda(), b.cuda())[-1])
        bwd.append(model(b.cuda(), a.cuda())[-1])
    return fwd, bwd


def main():
    frames = np.load(OUT / "frames.npz")["frames"]
    fwd, bwd = flows(frames)

    norm = {}
    for key, _ in METHODS:
        d = np.load(OUT / f"{key}.npz")["depths"]
        lo, hi = robust_range(d)
        norm[key] = torch.from_numpy(np.clip((d - lo) / (hi - lo), 0, 1)).float().cuda()[:, None]

    # One evaluation mask per pair, shared by both methods so they are scored on identical pixels.
    # Sky comes from the frame-wise model, which outputs exactly 0 there. VDA cannot be used: after
    # stitching windows it applies depth*scale + shift (clipped only below 0), which lifts its sky.
    sky_ref = norm["dav2_small_framewise"]
    masks = []
    for t in range(len(frames) - 1):
        f, b = fwd[t], bwd[t]
        b_at_f = warp(b, f)
        ok = ((f + b_at_f) ** 2).sum(1) < FB_TOL[0] * ((f**2).sum(1) + (b_at_f**2).sum(1)) + FB_TOL[1]
        inside = warp(torch.ones_like(sky_ref[:1]), f)[:, 0] > 0.999
        not_sky = (sky_ref[t, 0] > 0) | (warp(sky_ref[t + 1:t + 2], f)[0, 0] > 0)
        masks.append(ok[0] & inside[0] & not_sky)

    rows, per_pair = [], {"pair": np.arange(len(frames) - 1)}
    per_pair["mean_flow_px"] = [float(f.norm(dim=1).mean()) for f in fwd]  # context: camera / object motion per pair
    per_pair["eval_pixels"] = [int(m.sum()) for m in masks]
    sky_level = {"frame": np.arange(len(frames))}
    for key, name in METHODS:
        dn = norm[key]
        errs = []
        for t in range(len(frames) - 1):
            d_next = warp(dn[t + 1:t + 2], fwd[t])
            errs.append(float((dn[t, 0] - d_next[0, 0]).abs()[masks[t]].mean()))
        per_pair[name] = errs
        sky_level[name] = [float(dn[t, 0][sky_ref[t, 0] == 0].mean()) for t in range(len(frames))]
        rows.append({"method": name, "mean_abs_warp_error": float(np.mean(errs)), "median_over_pairs": float(np.median(errs)),
                     "max_over_pairs": float(np.max(errs)), "pairs": len(errs)})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/video_temporal_error.csv", index=False, float_format="%.4f")
    pd.DataFrame(per_pair).to_csv(ROOT / "results/video_temporal_error_per_pair.csv", index=False, float_format="%.5f")
    # Mean normalized value on sky pixels (frame-wise model == 0) per frame: exposes VDA's window-stitching offset
    pd.DataFrame(sky_level).to_csv(ROOT / "results/video_sky_level.csv", index=False, float_format="%.5f")
    print(df.to_string(index=False, float_format="%.4f"), flush=True)


if __name__ == "__main__":
    main()
