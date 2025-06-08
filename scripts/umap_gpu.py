import numpy as np
import time
import resource
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import cupy as cp
import os
import sys

from cuml.manifold import UMAP as cuUMAP
from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName, nvmlSystemGetDriverVersion
)

COMMON_PATH = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_PATH))
from visualizations import visualize

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH    = PROJECT_ROOT / "data"
RES_BASE     = PROJECT_ROOT / "results" / "umap_gpu"

use_normalized = True
use_pca        = False


if use_pca:
    if use_normalized:
        X = np.load(DATA_PATH / "X_pca_gpu_norm_90.npz")["matrix"].astype(np.float32)
        RES_DIR = RES_BASE / "pca_normalized"
    else:
        X = np.load(DATA_PATH / "X_pca_gpu_90.npz")["matrix"].astype(np.float32)
        RES_DIR = RES_BASE / "pca_non_normalized"
else:
    if use_normalized:
        X = np.load(DATA_PATH / "X_norm.npz")["X_norm"].astype(np.float32)
        RES_DIR = RES_BASE / "normalized"
    else:
        X = np.load(DATA_PATH / "X.npz")["X"].astype(np.float32)
        RES_DIR = RES_BASE / "non-normalized"

RES_DIR.mkdir(parents=True, exist_ok=True)

if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("UMAP input contains NaN/Inf")

X_gpu = cp.asarray(X)
free0, tot = cp.cuda.runtime.memGetInfo()
e0, e1 = cp.cuda.Event(), cp.cuda.Event()
e0.record()
wall0 = time.perf_counter()
cpu0  = resource.getrusage(resource.RUSAGE_SELF)


umap = cuUMAP(
    n_components=2,
    n_neighbors=10,
    n_epochs=1000,
    min_dist=0.1,
    spread = 3,
    init="spectral",
    metric="euclidean",
    output_type="cupy",
    verbose=True,
)
Y_gpu = umap.fit_transform(X_gpu)

e1.record(); e1.synchronize()
wall1 = time.perf_counter()
cpu1  = resource.getrusage(resource.RUSAGE_SELF)
free1, _ = cp.cuda.runtime.memGetInfo()

gpu_ms   = cp.cuda.get_elapsed_time(e0, e1)
wall_s   = wall1 - wall0
utime    = cpu1.ru_utime - cpu0.ru_utime
stime    = cpu1.ru_stime - cpu0.ru_stime
gpu_mem  = (free0 - free1) / 1e6

Y = Y_gpu.get()

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
gpu_name = nvmlDeviceGetName(h)
drv_ver  = nvmlSystemGetDriverVersion()

np.savez(DATA_PATH / f"X_umap_gpu_{'pca_' if use_pca else ''}{'norm' if use_normalized else 'raw'}.npz", matrix=Y)

with open(RES_DIR / "umap_gpu_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] cuML-UMAP\n")
    f.write(f"Wall time: {wall_s:.2f} s\n")
    f.write(f"CPU user {utime:.2f} s, sys {stime:.2f} s\n")
    f.write(f"GPU kernel {gpu_ms/1000:.3f} s\n")
    f.write(f"GPU mem: {gpu_mem:.1f} MB / {tot/1e6:.0f} MB\n")
    f.write(f"GPU: {gpu_name}, driver {drv_ver}\n")
    f.write(f"Points: {X.shape[0]}, In dims: {X.shape[1]}, Out dims: {Y.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")


def remove_outliers_percentile(Y, lower=0.001, upper=99.99):
    low = np.percentile(Y, lower, axis=0)
    high = np.percentile(Y, upper, axis=0)
    mask = np.all((Y >= low) & (Y <= high), axis=1)
    return Y[mask]

Y_filtered = remove_outliers_percentile(Y)

visualize(Y_filtered, "umap_gpu", RES_DIR)

print(" cuML-UMAP saved to", RES_DIR)
