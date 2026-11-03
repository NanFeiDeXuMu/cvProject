"""Part 1.2 / 2.2: frame-wise Depth Anything V2-Small vs Video Depth Anything-Small on the same frames.

Both models see exactly the same frames, read with VDA's own reader using the handout flags
(--max_len 100 --max_res 640). Outputs are relative (inverse) depth: larger = closer.
Raw float predictions are saved; no normalization happens here.

Usage: python -I src/video/run_video_depth.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from src.utils import ROOT, add_third_party
from src.utils.profiler import GpuProfiler, hardware_string

VITS = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
OUT = ROOT / "outputs/video"


def network_input_hw(h, w, size=518):
    """Network input size produced by the repos' shared 'lower_bound' Resize transform."""
    add_third_party("Video-Depth-Anything")
    from video_depth_anything.util.transform import Resize

    r = Resize(width=size, height=size, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound")
    in_w, in_h = r.get_size(w, h)
    return in_h, in_w


def run_framewise(frames, input_size, warmup=5):
    add_third_party("Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**VITS)
    model.load_state_dict(torch.load(ROOT / "checkpoints/depth_anything_v2_vits.pth", map_location="cpu"))
    model = model.cuda().eval()
    bgr = [np.ascontiguousarray(f[..., ::-1]) for f in frames]  # infer_image expects cv2 BGR
    for _ in range(warmup):
        model.infer_image(bgr[0], input_size)
    prof = GpuProfiler()
    depths = []
    for img in bgr:
        with prof:
            depths.append(model.infer_image(img, input_size))
    del model
    return np.stack(depths).astype(np.float32), prof.summary()


def run_video(frames, fps, input_size, fp32=False):
    add_third_party("Video-Depth-Anything")
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**VITS)
    model.load_state_dict(torch.load(ROOT / "checkpoints/video_depth_anything_vits.pth", map_location="cpu"), strict=True)
    model = model.cuda().eval()
    model.infer_video_depth(frames, fps, input_size=input_size, device="cuda", fp32=fp32)  # warm-up pass
    prof = GpuProfiler()
    with prof:
        depths, _ = model.infer_video_depth(frames, fps, input_size=input_size, device="cuda", fp32=fp32)
    s = prof.summary()
    s["ms_mean"] /= len(frames)  # one call covers the whole clip; report per frame
    s["ms_std"] = float("nan")
    del model
    return np.asarray(depths, dtype=np.float32), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4"))
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--max_res", type=int, default=640)
    ap.add_argument("--input_size", type=int, default=518)
    args = ap.parse_args()

    add_third_party("Video-Depth-Anything")
    from utils.dc_utils import read_video_frames  # VDA's reader, so both models get identical frames

    frames, fps = read_video_frames(args.video, args.max_len, -1, args.max_res)
    t, h, w = frames.shape[:3]
    print(f"{t} frames at {w}x{h}, {fps:.1f} fps (requested max_len={args.max_len})")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "frames.npz", frames=frames, fps=fps)

    in_h, in_w = network_input_hw(h, w, args.input_size)
    rows = []
    d_fw, s_fw = run_framewise(frames, args.input_size)
    np.savez_compressed(OUT / "dav2_small_framewise.npz", depths=d_fw)
    rows.append({"method": "Depth Anything V2-Small (frame-wise)", "precision": "FP32", **s_fw})
    torch.cuda.empty_cache()

    d_vd, s_vd = run_video(frames, fps, args.input_size)
    np.savez_compressed(OUT / "vda_small.npz", depths=d_vd)
    rows.append({"method": "Video Depth Anything-Small (offline)", "precision": "FP16 autocast", **s_vd})

    df = pd.DataFrame(rows)
    df.insert(2, "frames", t)
    df.insert(3, "frame_size", f"{w}x{h}")
    df.insert(4, "net_input", f"{in_w}x{in_h}")
    df["hardware"] = hardware_string()
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/video_efficiency.csv", index=False, float_format="%.2f")
    print(df.drop(columns=["hardware"]).to_markdown(index=False, floatfmt=".2f"))
    print(hardware_string())


if __name__ == "__main__":
    main()
