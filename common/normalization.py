import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings_data" / "embeddings"
PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data"

basis = np.load(RAW_DATA_PATH / "basis.npz")["matrix"]

np.savez_compressed(PROCESSED_DATA_PATH / "X.npz", X=basis)

row_sums = basis.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
X_norm = basis / row_sums

np.savez_compressed(PROCESSED_DATA_PATH / "X_norm.npz", X_norm=X_norm)

print("X.npz and X_norm.npz saved in", PROCESSED_DATA_PATH.resolve())
