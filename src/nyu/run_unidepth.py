"""Part 2.1: zero-shot metric depth with UniDepthV2-S on NYU Depth V2.

RGB only (no camera intrinsics given), repository defaults otherwise: default resolution bounds,
bilinear output interpolation, FP16 autocast inside infer(). No scale fitting of any kind.
Saves raw metric depth (already at 480x640) and the predicted intrinsics.

--gt_intrinsics is a DIAGNOSTIC only: it feeds NYU's calibrated camera to separate camera-estimation
error from depth error. It uses information the course protocol does not give the model, so its
numbers are never reported as the reproduction result.

Usage: python -I src/nyu/run_unidepth.py [--split test] [--gt_intrinsics]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import ROOT, add_third_party
from src.utils.metrics import NYU_SHAPE
from src.utils.profiler import GpuProfiler, hardware_string

MODEL_ID = "lpiccinelli/unidepth-v2-vits14"
# NYU toolbox camera_params.m (RGB camera)
NYU_K = np.array([[518.8579, 0, 325.5824], [0, 519.4696, 253.7362], [0, 0, 1]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gt_intrinsics", action="store_true", help="diagnostic: give the model NYU's calibrated camera")
    args = ap.parse_args()

    add_third_party("UniDepth")
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained(MODEL_ID)
    model.interpolation_mode = "bilinear"
    model = model.to("cuda").eval()

    rgb = np.load(ROOT / f"data/nyu/{args.split}_rgb.npy", mmap_mode="r")
    n = len(rgb)
    preds = np.empty((n, *NYU_SHAPE), np.float32)
    intrinsics = np.empty((n, 3, 3), np.float32)

    name = "unidepthv2_s_gtK" if args.gt_intrinsics else "unidepthv2_s"
    camera = torch.from_numpy(NYU_K) if args.gt_intrinsics else None
    to_tensor = lambda a: torch.from_numpy(np.array(a)).permute(2, 0, 1)  # copy out of the read-only mmap; uint8 C,H,W
    with torch.no_grad():
        for _ in range(args.warmup):
            model.infer(to_tensor(rgb[0]), camera)
        prof = GpuProfiler()
        for i in tqdm(range(n), desc=f"UniDepthV2-S {args.split}"):
            with prof:
                out = model.infer(to_tensor(rgb[i]), camera)
                depth = out["depth"].float()
                if depth.shape[-2:] != NYU_SHAPE:
                    depth = F.interpolate(depth, size=NYU_SHAPE, mode="bilinear", align_corners=False)
                preds[i] = depth[0, 0].cpu().numpy()
            intrinsics[i] = out["intrinsics"][0].float().cpu().numpy()

    # Network input size, computed with the repo's own preprocessing helpers
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor

    sc = model.shape_constraints
    _, padded = get_paddings(NYU_SHAPE, sc["ratio_bounds"])
    _, (net_h, net_w) = get_resize_factor(padded, [sc["pixels_min"], sc["pixels_max"]])

    out_dir = ROOT / "outputs/nyu"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_{args.split}_depth.npy", preds)
    np.save(out_dir / f"{name}_{args.split}_intrinsics.npy", intrinsics)
    stats = {
        "model": MODEL_ID,
        "label": "UniDepthV2-S + GT intrinsics (diagnostic)" if args.gt_intrinsics else "UniDepthV2-S (zero-shot, RGB only)",
        "split": args.split,
        "images": n,
        "input": ("RGB + NYU calibrated K" if args.gt_intrinsics else "RGB only") + ", 480x640 uint8; repo preprocessing (pad + resize to default pixel bounds)",
        "net_input": f"{net_w}x{net_h}",
        "precision": "FP16 autocast (repository default in infer)",
        "hardware": hardware_string(),
        **prof.summary(),
    }
    json.dump(stats, open(out_dir / f"{name}_{args.split}_profile.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
