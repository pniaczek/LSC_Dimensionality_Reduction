import numpy as np
import time
import resource
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys

from cuml.decomposition import PCA
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

if use_normalized:
    X = np.load(DATA_PATH / 'X_norm.npz')['X_norm'].astype(np.float32)
    RESULTS_DIR = PROJECT_ROOT / "results" / "pca_gpu" / "normalized"
else:
    X = np.load(DATA_PATH / 'X.npz')['X'].astype(np.float32)
    RESULTS_DIR = PROJECT_ROOT / "results" / "pca_gpu" / "non-normalized"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("Input contains NaN or Inf - cuML PCA cannot handle them.")

X_gpu = cp.asarray(X)
max_components = min(100, min(X_gpu.shape))

free_mem_before, total_mem = cp.cuda.runtime.memGetInfo()

evt_start = cp.cuda.Event(); evt_end = cp.cuda.Event()
evt_start.record()
wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

pca = PCA(n_components=max_components, svd_solver="jacobi")
X_pca_full = pca.fit_transform(X_gpu)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
evt_end.record(); evt_end.synchronize()
gpu_kernel_time_ms = cp.cuda.get_elapsed_time(evt_start, evt_end)
free_mem_after, _ = cp.cuda.runtime.memGetInfo()

explained_var = pca.explained_variance_ratio_ * 100
cumulative_var = cp.cumsum(explained_var).get()
explained_var = explained_var.get()
X_pca_full = X_pca_full.get()
var90_index = np.argmax(cumulative_var >= 90) + 1

utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime
gpu_mem_used_mb = (free_mem_before - free_mem_after) / 1e6

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
gpu_name = nvmlDeviceGetName(handle).decode()
driver_ver = nvmlSystemGetDriverVersion().decode()

X_pca_2d = X_pca_full[:, :2]
X_pca_90 = X_pca_full[:, :var90_index]
if use_normalized:
    np.savez(DATA_PATH / 'X_pca_gpu_norm_90.npz', matrix=X_pca_90)
else:
    np.savez(DATA_PATH / 'X_pca_gpu_90.npz', matrix=X_pca_90)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(explained_var) + 1), explained_var, marker='o', label="Explained variance")
plt.axvline(var90_index, color='r', linestyle='--', label=f'90% variance (PC {var90_index})')
plt.plot(var90_index, explained_var[var90_index - 1], 'ro')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance [%]")
plt.title("Explained Variance by GPU PCA Components (Top 100)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "explained_variance.png")

with open(RESULTS_DIR / "pca_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] PCA (AFDB, cuML GPU)\n")
    f.write(f"Explained variance (first 2): {explained_var[0]:.2f}%, {explained_var[1]:.2f}%\n")
    f.write(f"Total explained variance (100 comps): {cumulative_var[-1]:.2f}%\n")
    f.write(f"Components needed for >=90% variance: {var90_index}\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"GPU kernel time: {gpu_kernel_time_ms/1000:.3f} s\n")
    f.write(f"GPU memory used: {gpu_mem_used_mb:.1f} MB / {total_mem / 1e6:.1f} MB\n")
    f.write(f"GPU model: {gpu_name}, Driver: {driver_ver}\n")
    f.write(f"Points: {X.shape[0]}, Original dims: {X.shape[1]}, PCA dims saved: {X_pca_90.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

visualize(X_2d=X_pca_2d, method_name="pca_gpu", out_dir=RESULTS_DIR)

print(f"Plots and PCA (GPU) results saved in: {RESULTS_DIR.resolve()}")
