"""Side-by-side video and figures for the frame-wise vs video depth comparison.

Normalization rule (handout Section 6.2): never per frame. Each method gets ONE range, the
2nd-98th percentile of its own predictions over the whole clip, frozen for every frame.
Same palette for both (turbo, near = warm). Colors are relative depth, not a shared metric scale.

Usage: python -I src/video/make_video_figures.py [--row 200]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT
from src.utils.viz import colorize, robust_range

OUT = ROOT / "outputs/video"
FIG = ROOT / "figures"
CMAP = "turbo"
METHODS = [("dav2_small_framewise", "Depth Anything V2-S (frame-wise)"), ("vda_small", "Video Depth Anything-S")]
KEYFRAMES = [0, 23, 46, 69]
ZOOM = (slice(120, 260), slice(380, 600))  # beam crossings against the sky


def label_strip(width, text, height=30):
    strip = np.full((height, width, 3), 24, np.uint8)
    cv2.putText(strip, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return strip


def colorbar_strip(width, height=34):
    bar = colorize(np.tile(np.linspace(0, 1, width - 160), (12, 1)), 0, 1, CMAP)
    strip = np.full((height, width, 3), 24, np.uint8)
    strip[4:16, 80:80 + bar.shape[1]] = bar
    font, white = cv2.FONT_HERSHEY_SIMPLEX, (240, 240, 240)
    cv2.putText(strip, "far", (40, 15), font, 0.45, white, 1, cv2.LINE_AA)
    cv2.putText(strip, "near", (width - 74, 15), font, 0.45, white, 1, cv2.LINE_AA)
    cv2.putText(strip, "Relative depth. One normalization per method, frozen over all frames; colors are not a shared metric scale.",
                (80, 30), font, 0.4, white, 1, cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=200, help="image row used for the x-t slice")
    args = ap.parse_args()

    data = np.load(OUT / "frames.npz")
    frames, fps = data["frames"], float(data["fps"])
    t, h, w = frames.shape[:3]
    depths = {k: np.load(OUT / f"{k}.npz")["depths"] for k, _ in METHODS}
    ranges = {k: robust_range(d) for k, d in depths.items()}
    json.dump({k: {"vmin": r[0], "vmax": r[1], "rule": "2nd-98th percentile over all frames"} for k, r in ranges.items()},
              open(OUT / "normalization.json", "w"), indent=2)
    color = {k: np.stack([colorize(d[i], *ranges[k], CMAP) for i in range(t)]) for k, d in depths.items()}

    # 1) side-by-side video: RGB | frame-wise | video
    titles = ["RGB"] + [name for _, name in METHODS]
    header = np.concatenate([label_strip(w, s) for s in titles], 1)
    footer = colorbar_strip(w * 3)
    video_path = OUT / "davis_rollercoaster_side_by_side.mp4"
    with imageio.get_writer(video_path, fps=fps, codec="libx264", macro_block_size=1, ffmpeg_params=["-crf", "18"]) as wr:
        for i in range(t):
            body = np.concatenate([frames[i]] + [color[k][i] for k, _ in METHODS], 1)
            frame = np.concatenate([header, body, footer], 0)
            cv2.putText(frame, f"frame {i:02d}/{t - 1}", (w * 3 - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            wr.append_data(frame)

    # 2) key frames + zoomed crops
    fig, axes = plt.subplots(len(KEYFRAMES) + 1, 3, figsize=(12, 2.05 * (len(KEYFRAMES) + 1)))
    for r, i in enumerate(KEYFRAMES):
        for c, img in enumerate([frames[i]] + [color[k][i] for k, _ in METHODS]):
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(f"frame {i}", fontsize=9)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
        y, x = ZOOM
        axes[r, 0].add_patch(plt.Rectangle((x.start, y.start), x.stop - x.start, y.stop - y.start, fill=False, ec="w", lw=1))
    zi = KEYFRAMES[1]
    for c, img in enumerate([frames[zi]] + [color[k][zi] for k, _ in METHODS]):
        ax = axes[-1, c]
        ax.imshow(img[ZOOM], interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f"zoom, frame {zi}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "video_keyframes.png", dpi=130)
    plt.close(fig)

    # 3) x-t slice at one row: flicker shows up as horizontal streaks
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2))
    slices = [frames[:, args.row]] + [color[k][:, args.row] for k, _ in METHODS]
    for ax, img, title in zip(axes, slices, titles):
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_title(f"{title}: row y={args.row} over time", fontsize=10)
        ax.set_ylabel("frame")
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(FIG / "video_xt_slice.png", dpi=130)
    plt.close(fig)

    # 4) per-frame statistics of the raw output, each divided by its clip mean
    #    (sky is predicted as 0 and covers most pixels, so medians are 0 and unusable)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for (k, name), col in zip(METHODS, ["#c2410c", "#1d4ed8"]):
        d = depths[k].reshape(t, -1)
        clip_mean = d.mean()
        ax.plot(d.mean(1) / clip_mean, color=col, lw=1.8, label=f"{name}: mean")
        ax.plot(np.percentile(d, 90, 1) / clip_mean, color=col, lw=1, ls="--", label=f"{name}: 90th pct")
    ax.set_xlabel("frame")
    ax.set_ylabel("per-frame value / clip mean")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "video_scale_curve.png", dpi=130)
    plt.close(fig)

    # 5) flow-warped temporal error per frame pair, if temporal_consistency.py has been run
    per_pair = ROOT / "results/video_temporal_error_per_pair.csv"
    if per_pair.exists():
        err = pd.read_csv(per_pair)
        fig, ax = plt.subplots(figsize=(7, 3))
        for (_, name), col in zip(METHODS, ["#c2410c", "#1d4ed8"]):
            ax.plot(err["pair"], err[name], color=col, lw=1.6, label=f"{name} (mean {err[name].mean():.3f})")
        ax.set_xlabel("frame pair (t, t+1)")
        ax.set_ylabel("mean |d_t - warp(d_t+1)|\n(fraction of clip range)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "video_temporal_error.png", dpi=130)
        plt.close(fig)

    print(f"wrote {video_path.relative_to(ROOT)} and figures/video_*.png; ranges: {ranges}")


if __name__ == "__main__":
    main()
