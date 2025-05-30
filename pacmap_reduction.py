import numpy as np
import pacmap
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
import time
import resource
from datetime import datetime
from datashader.utils import export_image
import os


DATA_PATH = Path("data/embeddings_data/embeddings")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

n_neighbors_norm = 10
MN_ratio_norm = 1.3
FP_ratio_norm = 0.9

basis = np.load(DATA_PATH / 'basis.npz')['matrix']
X_norm = (basis.T / basis.sum(axis=1)).T

wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

embedding = pacmap.PaCMAP(n_components=2,
                           n_neighbors=n_neighbors_norm,
                           MN_ratio=MN_ratio_norm,
                           FP_ratio=FP_ratio_norm,
                           random_state=1)
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

plt.figure(figsize=(6, 5))
plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], alpha=0.05, c='gray', s=0.2, label='AFDB')
plt.legend(markerscale=10)
plt.title("PaCMAP 2D embedding")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pacmap_embedding.png", dpi=300)
plt.close()

df = pd.DataFrame(X_pacmap, columns=["x", "y"])
canvas = ds.Canvas(plot_width=1600, plot_height=1600)
agg = canvas.points(df, 'x', 'y')

def hex_cmap(name):
    return [mcolors.rgb2hex(cm.get_cmap(name)(i)) for i in np.linspace(0, 1, 256)]

variants = {
    "inferno_eqhist": {
        "cmap": hex_cmap("inferno"),
        "how": "eq_hist",
        "background": "white"
    },
    "inferno_log": {
        "cmap": hex_cmap("inferno"),
        "how": "log",
        "background": "white"
    },
    "coolwarm_eqhist_black": {
        "cmap": hex_cmap("coolwarm"),
        "how": "eq_hist",
        "background": "black"
    }
}

for name, settings in variants.items():
    img = tf.shade(agg, cmap=settings["cmap"], how=settings["how"], min_alpha=0)
    img = tf.dynspread(img, threshold=0.5, max_px=3)
    export_image(img, filename=str(RESULTS_DIR / f"pacmap_datashader_{name}"), background=settings["background"])

print(f"Zapisano wszystkie obrazy i log czasu w: {RESULTS_DIR.resolve()}")
