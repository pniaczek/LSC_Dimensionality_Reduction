import numpy as np
from openTSNE import TSNE
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
RESULTS_DIR = Path("results/tsne")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Wczytanie danych i normalizacja
basis = np.load(DATA_PATH / 'basis.npz')['matrix']
X_norm = (basis.T / basis.sum(axis=1)).T

# Pomiar czasu
wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

# Redukcja openTSNE (CPU, multicore)
embedding = TSNE(
    n_components=2,
    perplexity=30,
    initialization="pca",
    n_jobs=16,
    random_state=42,
    verbose=True
)
X_tsne = embedding.fit(X_norm)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

# Zapis logu
with open(RESULTS_DIR / "tsne_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] openTSNE\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X_tsne.shape[0]}, Dimensions: {X_tsne.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

# Wykres matplotlib
plt.figure(figsize=(6, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.05, c='gray', s=0.2, label='AFDB')
plt.legend(markerscale=10)
plt.title("openTSNE 2D embedding")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "tsne_embedding.png", dpi=300)
plt.close()

# Datashader
df = pd.DataFrame(X_tsne, columns=["x", "y"])
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
    export_image(img, filename=str(RESULTS_DIR / f"tsne_datashader_{name}"), background=settings["background"])

print(f"Zapisano openTSNE embedding i obrazy w: {RESULTS_DIR.resolve()}")
