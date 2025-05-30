import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datashader.utils import export_image
from pathlib import Path

RESULTS_DIR = Path("results/test/test_variants")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dane: 3 skupiska
np.random.seed(42)
n_points = 100_000
X1 = np.random.normal(loc=(-2, -2), scale=0.5, size=(n_points // 3, 2))
X2 = np.random.normal(loc=(2, 0), scale=0.5, size=(n_points // 3, 2))
X3 = np.random.normal(loc=(0, 3), scale=0.5, size=(n_points // 3, 2))
X = np.vstack([X1, X2, X3])
df = pd.DataFrame(X, columns=["x", "y"])

canvas = ds.Canvas(plot_width=800, plot_height=800)
agg = canvas.points(df, 'x', 'y')

# Lista wariantów: (nazwa, colormap, how, rozmycie, tło)
variants = [
    ("plasma_eqhist", "plasma", "eq_hist", None, "white"),
    ("plasma_eqhist_dynspread", "plasma", "eq_hist", "dynspread", "white"),
    ("inferno_log", "inferno", "log", None, "white"),
    ("inferno_log_spread", "inferno", "log", "spread", "white"),
    ("cividis_cbrt", "cividis", "cbrt", None, "white"),
    ("coolwarm_linear", "coolwarm", "linear", None, "black"),
    ("magma_eqhist_graybg", "magma", "eq_hist", "spread", "gray"),
    ("viridis_log_navybg", "viridis", "log", "dynspread", "navy"),
    ("blues_eqhist", "Blues", "eq_hist", None, "white"),
    ("custom_orange", None, "eq_hist", "dynspread", "white"),
]

# Własna mapa
custom_cmap = ["#fff5eb", "#fd8d3c", "#e6550d", "#a63603", "#7f2704"]

for name, cmap_name, how, blur, bg in variants:
    if cmap_name:
        cmap = [mcolors.rgb2hex(cm.get_cmap(cmap_name)(i)) for i in np.linspace(0, 1, 256)]
    else:
        cmap = custom_cmap

    img = tf.shade(agg, cmap=cmap, how=how, min_alpha=0)

    if blur == "spread":
        img = tf.spread(img, px=1)
    elif blur == "dynspread":
        img = tf.dynspread(img, threshold=0.5, max_px=3)

    export_image(img, filename=str(RESULTS_DIR / f"{name}"), background=bg)

print("Zapisano 10 wariantów do:", RESULTS_DIR)
