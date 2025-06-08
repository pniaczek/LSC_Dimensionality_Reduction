import numpy as np
import time
import resource
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys

from sklearn.utils import resample

from cuml.manifold import TSNE as cuMLTSNE
import cupy as cp
from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName, nvmlSystemGetDriverVersion
)

COMMON_PATH = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_PATH))
from visualizations import visualize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"

use_normalized = True
use_pca = True

if use_pca:
    if use_normalized:
        X = np.load(DATA_PATH / 'X_pca_gpu_norm_90.npz')['matrix'].astype(np.float32)
        RESULTS_DIR = PROJECT_ROOT / "results" / "tsne_gpu" / "pca_normalized"
    else:
        X = np.load(DATA_PATH / 'X_pca_gpu_90.npz')['matrix'].astype(np.float32)
        RESULTS_DIR = PROJECT_ROOT / "results" / "tsne_gpu" / "pca_non_normalized"
else:
    if use_normalized:
        X = np.load(DATA_PATH / 'X_norm.npz')['X_norm'].astype(np.float32)
        RESULTS_DIR = PROJECT_ROOT / "results" / "tsne_gpu" / "normalized"
    else:
        X = np.load(DATA_PATH / 'X.npz')['X'].astype(np.float32)
        RESULTS_DIR = PROJECT_ROOT / "results" / "tsne_gpu" / "non_normalized"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("Input contains NaN or Inf - cuML TSNE cannot handle them.")


X_gpu = cp.asarray(X)

free_mem_before, total_mem = cp.cuda.runtime.memGetInfo()

evt_start = cp.cuda.Event(); evt_end = cp.cuda.Event()
evt_start.record()
wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

tsne = cuMLTSNE(n_components=2, perplexity=30, n_iter=300, verbose=1, learning_rate=max(X.shape[0]/50, 1000), method="exact")
X_tsne = tsne.fit_transform(X_gpu)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
evt_end.record(); evt_end.synchronize()
gpu_kernel_time_ms = cp.cuda.get_elapsed_time(evt_start, evt_end)
free_mem_after, _ = cp.cuda.runtime.memGetInfo()

X_tsne = X_tsne.get()

utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime
gpu_mem_used_mb = (free_mem_before - free_mem_after) / 1e6

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
gpu_name = nvmlDeviceGetName(handle)
driver_ver = nvmlSystemGetDriverVersion()

txt_path = RESULTS_DIR / "tsne_gpu_time.txt"
with open(txt_path, "w") as f:
    f.write(f"[{datetime.now().isoformat()}] TSNE (cuML GPU)\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"GPU kernel time: {gpu_kernel_time_ms/1000:.3f} s\n")
    f.write(f"GPU memory used: {gpu_mem_used_mb:.1f} MB / {total_mem / 1e6:.1f} MB\n")
    f.write(f"GPU model: {gpu_name}, Driver: {driver_ver}\n")
    f.write(f"Points: {X.shape[0]}, Original dims: {X.shape[1]}, Output dims: {X_tsne.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

visualize(X_2d=X_tsne, method_name="tsne_gpu", out_dir=RESULTS_DIR)

print(f"cuML TSNE completed and saved in: {RESULTS_DIR.resolve()}")
