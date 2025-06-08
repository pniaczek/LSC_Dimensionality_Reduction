import numpy as np
import umap
import time
import resource
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys

COMMON_PATH = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_PATH))
from visualizations import visualize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_BASE = PROJECT_ROOT / "results" / "umap_cpu"

use_normalized = False
use_pca = False

if use_pca:
    if use_normalized:
        X = np.load(DATA_PATH / 'X_pca_gpu_norm_90.npz')['matrix'].astype(np.float32)
        RESULTS_DIR = RESULTS_BASE / "pca_normalized"
    else:
        X = np.load(DATA_PATH / 'X_pca_gpu_90.npz')['matrix'].astype(np.float32)
        RESULTS_DIR = RESULTS_BASE / "pca_non_normalized"
else:
    if use_normalized:
        X = np.load(DATA_PATH / 'X_norm.npz')['X_norm'].astype(np.float32)
        RESULTS_DIR = RESULTS_BASE / "normalized"
    else:
        X = np.load(DATA_PATH / 'X.npz')['X'].astype(np.float32)
        RESULTS_DIR = RESULTS_BASE / "non_normalized"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("Input contains NaN or Inf - UMAP cannot handle them.")

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

embedding = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
X_umap = embedding.fit_transform(X)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

with open(RESULTS_DIR / "umap_cpu_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] UMAP (CPU)")
    f.write(f"\nWall time: {wall_time:.2f} s")
    f.write(f"\nCPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s")
    f.write(f"\nPoints: {X.shape[0]}, Input dims: {X.shape[1]}, Output dims: {X_umap.shape[1]}")
    f.write(f"\nHostname: {os.uname().nodename}\n")

visualize(X_2d=X_umap, method_name="umap_cpu", out_dir=RESULTS_DIR)

plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1, alpha=0.6)
plt.title("UMAP CPU Embedding")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "umap_cpu_scatter.png")

print(f" UMAP CPU results saved in: {RESULTS_DIR.resolve()}")
