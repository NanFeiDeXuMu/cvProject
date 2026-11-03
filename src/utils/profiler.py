import time

import numpy as np
import torch


class GpuProfiler:
    """Per-call wall time (synchronized) and peak allocated GPU memory over a run."""

    def __init__(self, device="cuda"):
        self.device = device
        self.times = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self._t0)
        return False

    def summary(self, skip_first=0):
        t = np.array(self.times[skip_first:]) * 1000.0
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
        return {
            "calls": int(t.size),
            "ms_mean": float(t.mean()) if t.size else float("nan"),
            "ms_std": float(t.std()) if t.size else float("nan"),
            "peak_mem_MiB": float(peak),
        }


def hardware_string():
    if not torch.cuda.is_available():
        return "CPU"
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} ({p.total_memory / 2**30:.0f} GB), torch {torch.__version__}, CUDA {torch.version.cuda}"
