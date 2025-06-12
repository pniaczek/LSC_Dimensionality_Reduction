import datetime
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait
from dask_ml.decomposition import PCA
import dask.array as da
import numpy as np
from pathlib import Path
import sys
import time
import resource
import matplotlib.pyplot as plt
import os
import csv


COMMON_PATH = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_PATH))

from visualizations import visualize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"

use_normalized = False

if use_normalized:
    X = np.load(DATA_PATH / 'X_norm.npz')['X_norm']
    RESULTS_DIR = PROJECT_ROOT / "results" / "pca" / "normalized"
else:
    X = np.load(DATA_PATH / 'X.npz')['X']
    RESULTS_DIR = PROJECT_ROOT / "results" / "pca" / "non-normalized"

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

X_dask = da.from_array(X, chunks=(1000, -1))  # chunk by rows

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

# Setup Dask cluster
cluster = SLURMCluster(
    queue='plgrid',
    cores=8,
    memory='32GB',
    walltime='00:10:00',
    job_extra=["--account=plglscclass24-cpu"]
)
cluster.scale(jobs=2)

client = Client(cluster)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time_DaskSlurmCluster = wall_end - wall_start
cpu_total_DaskSlurmCluster = utime + stime

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

n_components = min(100, X.shape[0], X.shape[1])
pca = PCA(n_components=n_components)
X_pca_dask = pca.fit_transform(X_dask)  # lazy dask array
X_pca_res = X_pca_dask.compute()

explained_var = pca.explained_variance_ratio_ * 100
cumulative_var = np.cumsum(explained_var)
var90_index = np.argmax(cumulative_var >= 90) + 1

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

X_pca_2d = X_pca_res[:, :2]
X_pca_90 = X_pca_res[:, :var90_index]

if use_normalized:
    np.savez(DATA_PATH / 'X_pca_norm_90.npz', matrix=X_pca_90)
else:
    np.savez(DATA_PATH / 'X_pca_90.npz', matrix=X_pca_90)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(explained_var) + 1), explained_var, marker='o', label="Explained variance")
plt.axvline(var90_index, color='r', linestyle='--', label=f'90% variance (PC {var90_index})')
plt.plot(var90_index, explained_var[var90_index - 1], 'ro')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance [%]")
plt.title("Explained Variance by PCA Components (Top 100)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "explained_variance.png")

normalized = "norm" if use_normalized else "non-norm"

with open(RESULTS_DIR / f"pca_multinode_{normalized}.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] PCA (AFDB)\n")
    f.write(f"Explained variance (first 2): {explained_var[0]:.2f}%, {explained_var[1]:.2f}%\n")
    f.write(f"Total explained variance (100 comps): {cumulative_var[-1]:.2f}%\n")
    f.write(f"Components needed for >=90% variance: {var90_index}\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X.shape[0]}, Original dims: {X.shape[1]}, PCA dims saved: {X_pca_90.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")


headers = ["Explained_variance_%_1", "Explained_variance_2%", "Cumulative_variance_%", "Components_90_variance", "Wall_time_DaskSlurmCluster", "CPU_time_DaskSlurmCluster", "Wall_time", "CPU_time_total", "CPU_time_user", "CPU_time_sys",
           "Points", "Original_dims", "PCA_dim_saved", "Hostname"]
with open(RESULTS_DIR / f"pca_multinode_{normalized}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    data = [
        round(explained_var[0], 4),
        round(explained_var[1], 4),
        round(cumulative_var[-1], 4),
        var90_index,
        round(wall_time_DaskSlurmCluster, 4),
        round(cpu_total_DaskSlurmCluster, 4),
        round(wall_time, 4),
        round(cpu_total, 4),
        round(utime, 4),
        round(stime, 4),
        X.shape[0],
        X.shape[1],
        X_pca_90.shape[1],
        os.uname().nodename
    ]
    writer.writerow(data)

visualize(X_2d=X_pca_2d, method_name="pca", out_dir=RESULTS_DIR)

print(f"Plots and PCA results saved in: {RESULTS_DIR.resolve()}")

client.close()
cluster.close()
