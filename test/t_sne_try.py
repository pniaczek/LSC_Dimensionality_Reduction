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
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import os

# Ścieżki
RESULTS_DIR = Path("results/test/test_tsne")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Wczytaj MNIST
print("Pobieranie MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0

# Dla szybkości testu można wziąć podzbiór
X = X[:10000]  # np. tylko 10k punktów

# Pomiar czasu
wall_start = time.perf_counter()
cpu_start = resource.getrusage(resource.RUSAGE_SELF)

# t-SNE
embedding = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca", verbose=1)
X_tsne = embedding.fit_transform(X)

wall_end = time.perf_counter()
cpu_end = resource.getrusage(resource.RUSAGE_SELF)
utime = cpu_end.ru_utime - cpu_start.ru_utime
stime = cpu_end.ru_stime - cpu_start.ru_stime
wall_time = wall_end - wall_start
cpu_total = utime + stime

# Log czasu
with open(RESULTS_DIR / "tsne_time.txt", "w") as f:
    f.write(f"[{datetime.now().isoformat()}] t-SNE\n")
    f.write(f"Wall time: {wall_time:.2f} s\n")
    f.write(f"CPU times: user {utime:.2f} s, sys {stime:.2f} s, total {cpu_total:.2f} s\n")
    f.write(f"Points: {X_tsne.shape[0]}, Dimensions: {X_tsne.shape[1]}\n")
    f.write(f"Hostname: {os.uname().nodename}\n")

# Matplotlib
plt.figure(figsize=(6, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.05, c='gray', s=0.2, label='MNIST')
plt.legend(markerscale=10)
plt.title("t-SNE 2D embedding (MNIST)")
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
    export_image(img, filename=str(RESULTS_DIR / f"tsne_datashader_{name}"), background=settings["background"])

print(f"Zakończono wizualizacje t-SNE MNIST. Pliki w: {RESULTS_DIR.resolve()}")
