import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
import umap
import time
import resource
from datetime import datetime
from pathlib import Path
import os

# Ścieżki
RESULTS_DIR = Path("results/test/test_umap")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Wczytaj MNIST
print("Pobieranie MNIST...")
data = fetch_openml("mnist_784", version=1, as_frame=False)
X = data.data / 255.0  # normalizacja pikseli do [0,1]
y = data.target.astype(int)

# Zmniejsz zbior (np. do 5000)
X = X[:5000]
y = y[:5000]

# Pomiar czasu
wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
X_umap = reducer.fit_transform(X)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

# Log pomiaru czasu
with open(RESULTS_DIR / "umap_mnist_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] UMAP MNIST\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X_umap.shape[0]}, Dimensions: {X_umap.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

# Matplotlib wykres
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10", s=3, alpha=0.7)
plt.colorbar(label="Digit")
plt.title("UMAP MNIST 2D embedding")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mnist_umap_matplotlib.png", dpi=300)
plt.close()

# Datashader
df = pd.DataFrame(X_umap, columns=["x", "y"])
df["label"] = y
canvas = ds.Canvas(plot_width=1200, plot_height=1200)
agg = canvas.points(df, "x", "y")

def hex_cmap(name):
    return [mcolors.rgb2hex(cm.get_cmap(name)(i)) for i in np.linspace(0, 1, 256)]

variants = {
    "mnist_coolwarm_eqhist": {
        "cmap": hex_cmap("coolwarm"),
        "how": "eq_hist",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.5, "max_px": 3}
    },
    "mnist_inferno_log": {
        "cmap": hex_cmap("inferno"),
        "how": "log",
        "background": "white",
        "spread": tf.dynspread,
        "params": {"threshold": 0.3, "max_px": 4}
    },
    "mnist_blues_linear": {
        "cmap": hex_cmap("Blues"),
        "how": "linear",
        "background": "black",
        "spread": tf.dynspread,
        "params": {"threshold": 0.2, "max_px": 6}
    }
}

for name, settings in variants.items():
    img = tf.shade(agg, cmap=settings["cmap"], how=settings["how"], min_alpha=0)
    img = settings["spread"](img, **settings["params"])
    export_image(img, filename=str(RESULTS_DIR / name), background=settings["background"])

print(f"Zakonczono wizualizacje UMAP MNIST. Pliki w: {RESULTS_DIR.resolve()}")
