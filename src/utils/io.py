import re

import numpy as np


def read_pfm(path):
    """Read a PFM file into a float32 array (rows top-to-bottom, inf kept as-is)."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"{path}: not a PFM file")
        channels = 3 if header == "PF" else 1
        width, height = map(int, f.readline().decode("ascii").split())
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, dtype=endian + "f4", count=width * height * channels)
    shape = (height, width, 3) if channels == 3 else (height, width)
    # PFM stores scanlines bottom-to-top
    return np.flipud(data.reshape(shape)).astype(np.float32)


def read_middlebury_calib(path):
    """Parse a Middlebury 2014 calib.txt into a dict of floats / 3x3 matrices."""
    calib = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            if value.startswith("["):
                rows = [list(map(float, r.split())) for r in value.strip("[]").split(";")]
                calib[key] = np.array(rows, dtype=np.float64)
            else:
                calib[key] = float(value) if re.search(r"[.eE]", value) else int(value)
    return calib
