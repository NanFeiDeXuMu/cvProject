"""Export the NYU Depth V2 labeled split to .npy once, so experiments do not re-read the 2.8 GB .mat.

RGB = f["images"][i].transpose(2,1,0) (uint8, 480x640x3); depth = f["depths"][i].T (float32 meters,
the filled maps). splits.mat indices are 1-based (MATLAB) and converted to 0-based here.

Usage: python -I src/nyu/extract_nyu.py [--splits test train]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import scipy.io

from src.utils import ROOT

RAW = ROOT / "data/raw/nyu"
OUT = ROOT / "data/nyu"


def read_strings(f, key):
    return ["".join(map(chr, f[ref][()].flatten())) for ref in f[key][0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    split_mat = scipy.io.loadmat(RAW / "splits.mat")
    with h5py.File(RAW / "nyu_depth_v2_labeled.mat", "r") as f:
        scene_types = read_strings(f, "sceneTypes")
        scenes = read_strings(f, "scenes")
        for split in args.splits:
            idx = np.sort(split_mat[f"{split}Ndxs"].ravel().astype(int) - 1)  # 1-based -> 0-based
            rgb = np.empty((len(idx), 480, 640, 3), np.uint8)
            depth = np.empty((len(idx), 480, 640), np.float32)
            for s in range(0, len(idx), 64):
                chunk = idx[s:s + 64]  # h5py fancy indexing needs increasing indices
                rgb[s:s + len(chunk)] = f["images"][chunk].transpose(0, 3, 2, 1)
                depth[s:s + len(chunk)] = f["depths"][chunk].transpose(0, 2, 1)
            np.save(OUT / f"{split}_rgb.npy", rgb)
            np.save(OUT / f"{split}_depth.npy", depth)
            meta = [{"mat_index0": int(i), "scene_type": scene_types[i], "scene": scenes[i]} for i in idx]
            json.dump(meta, open(OUT / f"{split}_meta.json", "w"), indent=1)
            print(f"{split}: {len(idx)} images, depth range {depth.min():.3f}-{depth.max():.3f} m")


if __name__ == "__main__":
    main()
