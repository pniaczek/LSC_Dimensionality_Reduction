import numpy as np
import pacmap
import pickle
import matplotlib.pyplot as plt  # ← poprawiony import

from pathlib import Path

assert pacmap.__version__ == '0.7.2'

# Ścieżka do danych
DATA_PATH = Path("data/embeddings_data/embeddings")

# Wczytanie danych
basis = np.load(DATA_PATH / 'basis.npz')['matrix']

# Parametry PaCMAP (dla znormalizowanych danych)
n_neighbors_norm = 10
MN_ratio_norm = 1.3
FP_ratio_norm = 0.9

# Normalizacja wierszy
X_norm = (basis.T / basis.sum(axis=1)).T

# Redukcja
embedding = pacmap.PaCMAP(n_components=2,
                           n_neighbors=n_neighbors_norm,
                           MN_ratio=MN_ratio_norm,
                           FP_ratio=FP_ratio_norm,
                           random_state=1)

X_pacmap = embedding.fit_transform(X_norm, init="pca")

# Wykres i zapis
plt.figure(figsize=(6, 5))
plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], alpha=0.1, c='gray', s=1, label='AFDB + ESMAtlas + MIP')
plt.legend()
plt.title("PaCMAP 2D embedding")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()

# Zapis do pliku
plt.savefig("pacmap_embedding.png", dpi=300)
plt.close()

print("Zapisano: pacmap_embedding.png")
