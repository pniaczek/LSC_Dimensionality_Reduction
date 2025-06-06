import numpy as np
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "embeddings_data" / "embeddings"
basis = np.load(data_path / "basis.npz")["matrix"]

row_sums = basis.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
X_norm = basis / row_sums

np.savez_compressed(data_path / "X_norm.npz", X_norm=X_norm)
