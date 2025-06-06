import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from datashader.utils import export_image


def hex_cmap(name):
    return [mcolors.rgb2hex(cm.get_cmap(name)(i)) for i in np.linspace(0, 1, 256)]


VARIANTS = {
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


def plot_matplotlib(X_2d, method_name, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.05, c='gray', s=0.2, label='AFDB')
    plt.legend(markerscale=10)
    plt.title(f"{method_name.upper()} 2D embedding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(out_dir / f"{method_name}_embedding.png", dpi=300)
    plt.close()


def plot_datashader(X_2d, method_name, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True) 
    df = pd.DataFrame(X_2d, columns=["x", "y"])
    canvas = ds.Canvas(plot_width=1600, plot_height=1600)
    agg = canvas.points(df, 'x', 'y')
    for name, settings in VARIANTS.items():
        img = tf.shade(agg, cmap=settings["cmap"], how=settings["how"], min_alpha=0)
        img = settings["spread"](img, **settings["params"])
        export_image(img, filename=str(out_dir / f"{method_name}_datashader_{name}"), background=settings["background"])


def visualize(X_2d: np.ndarray, method_name: str, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "results" / method_name.lower()

    plot_matplotlib(X_2d, method_name, out_dir)
    plot_datashader(X_2d, method_name, out_dir)
    print(f"Plots were generated and saved in: {out_dir.resolve()}")
