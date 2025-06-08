import numpy as np
import time
import resource
from datetime import datetime
from pathlib import Path
import os
import sys

from openTSNE import TSNE

COMMON_PATH = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_PATH))
from visualizations import visualize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_BASE = PROJECT_ROOT / "results" / "tsne_cpu"

use_normalized = False
use_pca =  True

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
    raise ValueError("Input contains NaN or Inf - openTSNE cannot handle them.")

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    initialization="pca",
    n_jobs=-1,
    random_state=42,
    verbose=True
)
X_tsne = tsne.fit(X)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

with open(RESULTS_DIR / "tsne_cpu_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] TSNE (openTSNE CPU)\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X.shape[0]}, Input dims: {X.shape[1]}, Output dims: {X_tsne.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

visualize(X_2d=X_tsne, method_name="tsne_cpu", out_dir=RESULTS_DIR)

print(f"openTSNE results saved in: {RESULTS_DIR.resolve()}")
