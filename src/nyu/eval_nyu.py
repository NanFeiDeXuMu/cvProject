"""Evaluate saved NYU predictions with the course protocol and draw the qualitative figures.

Protocol (handout 6.2): predictions at 480x640, crop [45:471, 41:601], GT in [0.001, 10] m,
predictions clipped to that range, per-image AbsRel / RMSE / delta1 averaged over all 654 test
images, no scale alignment. The per-image median(GT/pred) ratio and predicted focal length are
stored as diagnostics only; they never change the reported metrics.

Usage: python -I src/nyu/eval_nyu.py [--name unidepthv2_s]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT
from src.utils.metrics import NYU_CROP, NYU_MAX_DEPTH, NYU_MIN_DEPTH, depth_metrics, nyu_eval_mask
from src.utils.viz import colorize

NYU_FX = 518.8579  # NYU toolbox camera_params.m, fx_rgb
GREY = (150, 150, 150)
FIG = ROOT / "figures"
RES = ROOT / "results"


def abs_rel_map(pred, gt, mask):
    p = np.clip(pred, NYU_MIN_DEPTH, NYU_MAX_DEPTH)
    return np.where(mask, np.abs(p - gt) / np.maximum(gt, NYU_MIN_DEPTH), np.nan)


def panel_row(rgb, gt, pred, mask):
    lo, hi = np.percentile(gt[mask], [1, 99])
    err = abs_rel_map(pred, gt, mask)
    return [
        rgb,
        colorize(gt, lo, hi, "turbo_r", ~mask, GREY),
        colorize(pred, lo, hi, "turbo_r"),
        colorize(err, 0, 0.3, "magma", ~mask, GREY),
    ], (lo, hi)


def grid_figure(rows, labels, path, titles=("RGB (eval crop boxed)", "GT depth", "UniDepthV2-S depth", "AbsRel 0-0.3 (grey = not evaluated)")):
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 2.55 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (imgs, label) in enumerate(zip(rows, labels)):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        y, x = NYU_CROP
        axes[r, 0].add_patch(plt.Rectangle((x.start, y.start), x.stop - x.start, y.stop - y.start, fill=False, ec="w", lw=1, ls="--"))
        axes[r, 0].set_ylabel(label, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def strongest_edge_window(gt, mask, h=120, w=160):
    """Top-left corner of the window inside the crop with the most GT depth-discontinuity energy."""
    gy, gx = np.gradient(np.where(mask, gt, np.nan))
    energy = np.nan_to_num(np.hypot(gx, gy))
    best, corner = -1, (NYU_CROP[0].start, NYU_CROP[1].start)
    for y in range(NYU_CROP[0].start, NYU_CROP[0].stop - h, 20):
        for x in range(NYU_CROP[1].start, NYU_CROP[1].stop - w, 20):
            e = energy[y:y + h, x:x + w].sum()
            if e > best:
                best, corner = e, (y, x)
    return corner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="unidepthv2_s")
    ap.add_argument("--no-figures", dest="figures", action="store_false")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    gt_all = np.load(ROOT / "data/nyu/test_depth.npy", mmap_mode="r")
    rgb_all = np.load(ROOT / "data/nyu/test_rgb.npy", mmap_mode="r")
    meta = json.load(open(ROOT / "data/nyu/test_meta.json"))
    pred_all = np.load(ROOT / f"outputs/nyu/{args.name}_test_depth.npy", mmap_mode="r")
    K = np.load(ROOT / f"outputs/nyu/{args.name}_test_intrinsics.npy")
    profile = json.load(open(ROOT / f"outputs/nyu/{args.name}_test_profile.json"))
    assert len(gt_all) == len(pred_all) == 654, (len(gt_all), len(pred_all))

    rows = []
    for i in range(len(gt_all)):
        gt, pred = np.asarray(gt_all[i]), np.asarray(pred_all[i])
        mask = nyu_eval_mask(gt)
        m = depth_metrics(pred, gt, mask)
        ratio = float(np.median(gt[mask] / np.clip(pred[mask], NYU_MIN_DEPTH, None)))
        rows.append({"i": i, **meta[i], **m, "diag_median_gt_over_pred": ratio, "fx_pred": float(K[i, 0, 0])})
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"nyu_{args.name}_per_image.csv", index=False, float_format="%.5f")

    summary = {
        "method": profile["label"],
        "images": len(df),
        "abs_rel": df.abs_rel.mean(),
        "rmse": df.rmse.mean(),
        "delta1": df.delta1.mean(),
        "net_input": profile["net_input"],
        "precision": profile["precision"],
        "ms_per_image": profile["ms_mean"],
        "peak_mem_MiB": profile["peak_mem_MiB"],
        "hardware": profile["hardware"],
    }
    pd.DataFrame([summary]).to_csv(RES / f"nyu_{args.name}_summary.csv", index=False, float_format="%.4f")
    by_scene = df.groupby("scene_type").agg(n=("i", "size"), abs_rel=("abs_rel", "mean"), rmse=("rmse", "mean"),
                                            delta1=("delta1", "mean"), median_scale_ratio=("diag_median_gt_over_pred", "median"))
    by_scene.sort_values("n", ascending=False).to_csv(RES / f"nyu_{args.name}_by_scene_type.csv", float_format="%.4f")

    worst = df.nlargest(3, "abs_rel")
    print(pd.DataFrame([summary]).drop(columns=["hardware"]).to_markdown(index=False, floatfmt=".4f"))
    print(f"\nDiagnostics: median per-image GT/pred = {df.diag_median_gt_over_pred.median():.3f} "
          f"(IQR {df.diag_median_gt_over_pred.quantile(.25):.3f}-{df.diag_median_gt_over_pred.quantile(.75):.3f}); "
          f"median fx_pred = {df.fx_pred.median():.1f} px vs NYU {NYU_FX:.1f}; "
          f"corr(fx error, scale ratio) = {np.corrcoef(df.fx_pred / NYU_FX - 1, df.diag_median_gt_over_pred)[0, 1]:.3f}")
    print("worst 3:", worst[["i", "scene_type", "abs_rel"]].to_dict("records"))
    if not args.figures:
        return

    # Qualitative: four evenly spaced test images
    picks = np.linspace(0, len(df) - 1, 4).astype(int)
    rows_img, labels = [], []
    for i in picks:
        gt, pred = np.asarray(gt_all[i]), np.asarray(pred_all[i])
        imgs, (lo, hi) = panel_row(np.asarray(rgb_all[i]), gt, pred, nyu_eval_mask(gt))
        rows_img.append(imgs)
        labels.append(f"#{i} {meta[i]['scene_type']}\nAbsRel {df.abs_rel[i]:.3f}, {lo:.1f}-{hi:.1f} m")
    grid_figure(rows_img, labels, FIG / "nyu_qualitative.png")

    # Failure cases: the three highest-AbsRel test images
    rows_img, labels = [], []
    for i in worst.i:
        gt, pred = np.asarray(gt_all[i]), np.asarray(pred_all[i])
        imgs, (lo, hi) = panel_row(np.asarray(rgb_all[i]), gt, pred, nyu_eval_mask(gt))
        rows_img.append(imgs)
        labels.append(f"#{i} {meta[i]['scene_type']}\nAbsRel {df.abs_rel[i]:.3f}, med GT/pred {df.diag_median_gt_over_pred[i]:.2f}")
    grid_figure(rows_img, labels, FIG / "nyu_worst3.png")

    # Boundary zoom on the median-AbsRel image, around its strongest depth discontinuities
    i = int(df.iloc[(df.abs_rel - df.abs_rel.median()).abs().argsort().iloc[0]].i)
    gt, pred, rgb = np.asarray(gt_all[i]), np.asarray(pred_all[i]), np.asarray(rgb_all[i])
    mask = nyu_eval_mask(gt)
    (y, x), (h, w) = strongest_edge_window(gt, mask), (120, 160)
    imgs, _ = panel_row(rgb, gt, pred, mask)
    zoom = [im[y:y + h, x:x + w] for im in imgs]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.7))
    for ax, im, t in zip(axes, zoom, ("RGB", "GT", "UniDepthV2-S", "AbsRel 0-0.3")):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(f"{t} (zoom, #{i})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "nyu_boundary_zoom.png", dpi=130)
    plt.close(fig)

    # Diagnostics for RQ2: per-image scale ratio and predicted focal length
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(df.diag_median_gt_over_pred, bins=40, color="#2a50c6")
    axes[0].axvline(1.0, color="k", lw=1)
    axes[0].set_xlabel("per-image median(GT / pred)  [diagnostic]")
    axes[0].set_ylabel("test images")
    axes[1].scatter(df.fx_pred / NYU_FX - 1, df.diag_median_gt_over_pred, s=6, color="#2a50c6", alpha=0.6)
    axes[1].axhline(1.0, color="k", lw=0.8), axes[1].axvline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("predicted fx / NYU fx - 1")
    axes[1].set_ylabel("median(GT / pred)")
    fig.tight_layout()
    fig.savefig(FIG / "nyu_scale_diagnostics.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
