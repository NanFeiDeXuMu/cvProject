import numpy as np

# Course protocol for NYU Depth V2 (Section 6.2 of the handout)
NYU_SHAPE = (480, 640)
NYU_CROP = (slice(45, 471), slice(41, 601))
NYU_MIN_DEPTH, NYU_MAX_DEPTH = 1e-3, 10.0


def nyu_eval_mask(gt):
    """Valid pixels inside the evaluation crop with GT in [0.001, 10] m."""
    assert gt.shape == NYU_SHAPE, gt.shape
    mask = np.zeros(NYU_SHAPE, dtype=bool)
    mask[NYU_CROP] = True
    return mask & (gt >= NYU_MIN_DEPTH) & (gt <= NYU_MAX_DEPTH)


def depth_metrics(pred, gt, mask):
    """AbsRel, RMSE and delta1 on masked pixels. Predictions are clipped to the evaluation range; no scale alignment."""
    p = np.clip(pred[mask].astype(np.float64), NYU_MIN_DEPTH, NYU_MAX_DEPTH)
    g = gt[mask].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def stereo_metrics(pred_disp, pred_valid, gt_disp, nocc_mask=None):
    """Middlebury exercise metrics on finite, positive GT disparities."""
    gt_valid = np.isfinite(gt_disp) & (gt_disp > 0)
    both = gt_valid & pred_valid
    err = np.abs(pred_disp[both] - gt_disp[both])
    out = {
        "mae_px": float(err.mean()) if err.size else float("nan"),
        "valid_frac": float(both.sum() / max(gt_valid.sum(), 1)),
        "bad2": float((err > 2.0).mean()) if err.size else float("nan"),
    }
    if nocc_mask is not None:
        both_nocc = both & nocc_mask
        out["mae_px_nocc"] = float(np.abs(pred_disp[both_nocc] - gt_disp[both_nocc]).mean())
    return out
