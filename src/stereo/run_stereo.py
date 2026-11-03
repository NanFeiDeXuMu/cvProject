"""Part 1.1: classical stereo on the 15 Middlebury v3 trainingQ pairs.

Runs StereoBM, StereoSGBM and StereoSGBM + explicit left-right check, converts disparity to
metric depth with each scene's calib.txt, and evaluates against disp0GT.pfm.

Invalid-prediction rule: after dividing OpenCV's fixed-point output by 16, a pixel is invalid
if d <= 0 (OpenCV writes minDisparity - 1 = -1 for unmatched / speckle-filtered pixels).
With the LR check, a pixel is also invalid if |d_L(x) - d_R(x - d_L(x))| > 1 px.

Usage: python -I src/stereo/run_stereo.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT
from src.utils.io import read_middlebury_calib, read_pfm
from src.utils.metrics import stereo_metrics
from src.utils.viz import colorize

DATA = ROOT / "data/middlebury/MiddEval3/trainingQ"
OUT = ROOT / "outputs/stereo"
FIG = ROOT / "outputs/stereo/figures"  # per-scene panels (regenerable); report figure goes to figures/
RES = ROOT / "results"
LR_THRESHOLD_PX = 1.0
OVERVIEW_SCENES = ["Adirondack", "Jadeplant", "PianoL", "Shelves"]


def make_matcher(kind, num_disp):
    if kind == "BM":
        m = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
        m.setUniquenessRatio(15)
        m.setSpeckleWindowSize(100)
        m.setSpeckleRange(32)  # BM filters the x16 fixed-point map, so 32 = 2 px
        return m
    block = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block**2,
        P2=32 * 3 * block**2,
        disp12MaxDiff=-1,  # LR check is done explicitly in the "SGBM+LR" variant
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(matcher, left, right):
    return matcher.compute(left, right).astype(np.float32) / 16.0


def lr_consistent(disp_l, disp_r):
    """Keep left pixels whose disparity agrees with the right map at x - d."""
    h, w = disp_l.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    xr = np.round(xs - disp_l).astype(int)
    inside = (xr >= 0) & (xr < w)
    d_r = np.full_like(disp_l, np.nan)
    rows = np.arange(h)[:, None].repeat(w, 1)
    d_r[inside] = disp_r[rows[inside], xr[inside]]
    return inside & np.isfinite(d_r) & (d_r > 0) & (np.abs(disp_l - d_r) <= LR_THRESHOLD_PX)


def run_variant(kind, im0, im1, num_disp):
    color = kind != "BM"
    left = im0 if color else cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    right = im1 if color else cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    matcher = make_matcher(kind, num_disp)
    t0 = time.perf_counter()
    disp = compute_disparity(matcher, left, right)
    valid = disp > 0
    if kind == "SGBM+LR":
        # Right-view disparity: match the mirrored pair (right as reference), then mirror back
        disp_r = compute_disparity(matcher, np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))[:, ::-1]
        valid &= lr_consistent(disp, disp_r)
    ms = (time.perf_counter() - t0) * 1000.0
    return disp, valid, ms


def disparity_to_depth(disp, calib):
    """Z[m] = f * baseline[mm] / (d + doffs) / 1000."""
    return calib["cam0"][0, 0] * calib["baseline"] / (disp + calib["doffs"]) / 1000.0


def scene_figure(scene, im0, gt, calib, disp, valid, path):
    vmin, vmax = calib["vmin"], calib["vmax"]
    gt_valid = np.isfinite(gt) & (gt > 0)
    err = np.where(valid & gt_valid, np.abs(disp - gt), np.nan)
    depth = disparity_to_depth(disp, calib)
    zmin, zmax = disparity_to_depth(np.float64(vmax), calib), disparity_to_depth(np.float64(vmin), calib)
    panels = [
        ("Left image", cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)),
        ("GT disparity", colorize(gt, vmin, vmax, "turbo", ~gt_valid)),
        ("SGBM disparity (black = invalid)", colorize(disp, vmin, vmax, "turbo", ~valid)),
        ("|error| px, 0-5 (grey = not evaluated)", colorize(err, 0, 5, "magma", ~np.isfinite(err), (150, 150, 150))),
        (f"SGBM depth, {zmin:.2f}-{zmax:.2f} m", colorize(depth, zmin, zmax, "turbo_r", ~valid)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.2))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(scene, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overview_figure(cache, path):
    cols = ["Left image", "GT", "BM", "SGBM", "SGBM+LR", "SGBM |error| (0-5 px)"]
    fig, axes = plt.subplots(len(OVERVIEW_SCENES), len(cols), figsize=(2.9 * len(cols), 2.2 * len(OVERVIEW_SCENES)))
    for r, scene in enumerate(OVERVIEW_SCENES):
        c = cache[scene]
        vmin, vmax = c["calib"]["vmin"], c["calib"]["vmax"]
        gt_valid = np.isfinite(c["gt"]) & (c["gt"] > 0)
        d, v = c["SGBM"]
        err = np.where(v & gt_valid, np.abs(d - c["gt"]), np.nan)
        imgs = [
            cv2.cvtColor(c["im0"], cv2.COLOR_BGR2RGB),
            colorize(c["gt"], vmin, vmax, "turbo", ~gt_valid),
            *[colorize(c[k][0], vmin, vmax, "turbo", ~c[k][1]) for k in ("BM", "SGBM", "SGBM+LR")],
            colorize(err, 0, 5, "magma", ~np.isfinite(err), (150, 150, 150)),
        ]
        for col, img in enumerate(imgs):
            ax = axes[r, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(scene, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    for d in (OUT, FIG, RES):
        d.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    assert len(scenes) == 15, scenes
    rows, cache = [], {}
    for scene in scenes:
        sd = DATA / scene
        im0, im1 = cv2.imread(str(sd / "im0.png")), cv2.imread(str(sd / "im1.png"))
        calib = read_middlebury_calib(sd / "calib.txt")
        gt = read_pfm(sd / "disp0GT.pfm")
        nocc = cv2.imread(str(sd / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255
        num_disp = int(np.ceil(calib["ndisp"] / 16.0) * 16)
        gt_valid = np.isfinite(gt) & (gt > 0)
        gt_depth = disparity_to_depth(np.where(gt_valid, gt, np.nan), calib)
        cache[scene] = {"im0": im0, "gt": gt, "calib": calib}
        for kind in ("BM", "SGBM", "SGBM+LR"):
            disp, valid, ms = run_variant(kind, im0, im1, num_disp)
            depth = disparity_to_depth(disp, calib)
            m = stereo_metrics(disp, valid, gt, nocc)
            both = valid & gt_valid
            m["depth_abs_rel"] = float(np.mean(np.abs(depth[both] - gt_depth[both]) / gt_depth[both]))
            rows.append({"scene": scene, "method": kind, **m, "time_ms": ms, "num_disp": num_disp})
            cache[scene][kind] = (disp, valid)
            vd = OUT / kind
            vd.mkdir(exist_ok=True)
            np.save(vd / f"{scene}_disp.npy", np.where(valid, disp, np.nan).astype(np.float32))
            np.save(vd / f"{scene}_depth_m.npy", np.where(valid, depth, np.nan).astype(np.float32))
        scene_figure(scene, im0, gt, calib, *cache[scene]["SGBM"], FIG / f"{scene}.png")
        print(f"{scene:12s} " + "  ".join(f"{r['method']}: {r['mae_px']:.2f}px/{r['valid_frac']:.1%}" for r in rows[-3:]))

    df = pd.DataFrame(rows)
    df.to_csv(RES / "stereo_per_scene.csv", index=False, float_format="%.4f")
    summary = df.groupby("method", sort=False)[["mae_px", "mae_px_nocc", "valid_frac", "bad2", "depth_abs_rel", "time_ms"]].mean()
    summary.to_csv(RES / "stereo_summary.csv", float_format="%.4f")
    overview_figure(cache, ROOT / "figures/stereo_overview.png")
    print("\nMean over 15 trainingQ scenes:")
    print(summary.to_markdown(floatfmt=".3f"))


if __name__ == "__main__":
    main()
