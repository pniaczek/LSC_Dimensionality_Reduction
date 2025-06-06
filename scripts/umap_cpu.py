import numpy as np
import umap
import time
import resource
from datetime import datetime
from pathlib import Path
import os
import sys

COMMON_PATH = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_PATH))

from visualizations import visualize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data/embeddings_data/embeddings"
RESULTS_DIR = PROJECT_ROOT / "results" / "umap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

basis = np.load(DATA_PATH / 'basis.npz')['matrix']
X_norm = (basis.T / basis.sum(axis=1)).T

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

embedding = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
X_umap = embedding.fit_transform(X_norm)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

with open(RESULTS_DIR / "umap_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] UMAP\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X_umap.shape[0]}, Dimensions: {X_umap.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

visualize(X_2d=X_umap, method_name="umap", out_dir=RESULTS_DIR)

print(f"Plots and output saved in: {RESULTS_DIR.resolve()}")
