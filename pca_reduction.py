import numpy as np
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
from sklearn.decomposition import PCA
import os

DATA_PATH = Path("data/embeddings_data/embeddings")
RESULTS_DIR = Path("results/pca")
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

plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.05, c='gray', s=0.2, label='AFDB')
plt.legend(markerscale=10)
plt.title(f"PCA 2D embedding\nExplained variance: {explained_var[0]:.1f}%, {explained_var[1]:.1f}%")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pca_embedding.png", dpi=300)
plt.close()

df = pd.DataFrame(X_pca, columns=["x", "y"])
canvas = ds.Canvas(plot_width=1600, plot_height=1600)
agg = canvas.points(df, 'x', 'y')

def hex_cmap(name):
    return [mcolors.rgb2hex(cm.get_cmap(name)(i)) for i in np.linspace(0, 1, 256)]

variants = {
    "inferno_eqhist": {
        "cmap": hex_cmap("inferno"),
        "how": "eq_hist",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 3}
    },
    "inferno_log": {
        "cmap": hex_cmap("inferno"),
        "how": "log",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 3}
    },
    "coolwarm_eqhist_black": {
        "cmap": hex_cmap("coolwarm"),
        "how": "eq_hist",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 18}
    },
    "coolwarm_eqhist_black_blurred": {
        "cmap": hex_cmap("coolwarm"),
        "how": "eq_hist",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 25}
    }
}

for name, settings in variants.items():
    img = tf.shade(agg, cmap=settings["cmap"], how=settings["how"], min_alpha=0)
    img = settings["spread"](img, **settings["params"])
    export_image(img, filename=str(RESULTS_DIR / f"pca_datashader_{name}"), background=settings["background"])

print(f"PCA zakończone. Wyniki w: {RESULTS_DIR.resolve()}")
