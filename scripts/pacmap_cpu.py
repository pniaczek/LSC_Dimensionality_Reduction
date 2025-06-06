import numpy as np
import pacmap
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
RESULTS_DIR = PROJECT_ROOT / "results" / "pacmap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

n_neighbors_norm = 10
MN_ratio_norm = 1.3
FP_ratio_norm = 0.9

basis = np.load(DATA_PATH / 'basis.npz')['matrix']
X_norm = (basis.T / basis.sum(axis=1)).T

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

embedding = pacmap.PaCMAP(
    n_components=2,
    n_neighbors=n_neighbors_norm,
    MN_ratio=MN_ratio_norm,
    FP_ratio=FP_ratio_norm,
    random_state=1
)
X_pacmap = embedding.fit_transform(X_norm, init="pca")

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

with open(RESULTS_DIR / "pacmap_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] PaCMAP\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X_pacmap.shape[0]}, Dimensions: {X_pacmap.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

visualize(X_2d=X_pacmap, method_name="pacmap", out_dir=RESULTS_DIR)

print(f"Plots and output saved in: {RESULTS_DIR.resolve()}")
