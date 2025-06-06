import numpy as np
import time
import resource
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA
import os
import sys

COMMON_PATH = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_PATH))

from visualizations import visualize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data/embeddings_data/embeddings"
RESULTS_DIR = PROJECT_ROOT / "results" / "pca"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

basis = np.load(DATA_PATH / 'basis.npz')['matrix']
X_norm = (basis.T / basis.sum(axis=1)).T

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)
explained_var = pca.explained_variance_ratio_ * 100

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

with open(RESULTS_DIR / "pca_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] PCA (AFDB + ESMAtlas + MIP)\n")
    f.write(f"Explained variance: {explained_var[0]:.2f}%, {explained_var[1]:.2f}%\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X_pca.shape[0]}, Dimensions: {X_pca.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

visualize(X_2d=X_pca, method_name="pca", out_dir=RESULTS_DIR)

print(f"Plots and output saved in: {RESULTS_DIR.resolve()}")
