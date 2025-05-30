import numpy as np
import umap
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


plt.figure(figsize=(6, 5))
plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.05, c='gray', s=0.2, label='AFDB')
plt.legend(markerscale=10)
plt.title("UMAP 2D embedding")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "umap_embedding.png", dpi=300)
plt.close()

df = pd.DataFrame(X_umap, columns=["x", "y"])
canvas = ds.Canvas(plot_width=1600, plot_height=1600)
agg = canvas.points(df, 'x', 'y')

def hex_cmap(name):
    return [mcolors.rgb2hex(cm.get_cmap(name)(i)) for i in np.linspace(0, 1, 256)]

variants = {
    "inferno_eqhist": {
        "cmap": hex_cmap("inferno"),
        "how": "eq_hist",
        "background": "white",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 3}
    },
    "inferno_log": {
        "cmap": hex_cmap("inferno"),
        "how": "log",
        "background": "white",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 3}
    },
    "coolwarm_eqhist_black": {
        "cmap": hex_cmap("coolwarm"),
        "how": "eq_hist",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 3}
    },
    "coolwarm_eqhist_black_blurred": {
        "cmap": hex_cmap("coolwarm"),
        "how": "eq_hist",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.2, "max_px": 6}
    }
}

for name, settings in variants.items():
    img = tf.shade(agg, cmap=settings["cmap"], how=settings["how"], min_alpha=0)
    img = settings["spread"](img, **settings["params"])
    export_image(img, filename=str(RESULTS_DIR / f"umap_datashader_{name}"), background=settings["background"])

print(f"Zapisano UMAP embedding i obrazy w: {RESULTS_DIR.resolve()}")
