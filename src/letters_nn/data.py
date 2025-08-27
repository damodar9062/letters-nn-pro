from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

letter_map = ["a","e","g","i","l","n","o","r","t","u"]

def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    y = df.iloc[:,0].to_numpy(dtype=int)
    X = df.iloc[:,1:].to_numpy(dtype=float)
    # normalize to [0,1] if values are 0/1 it's no-op
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    return X, y

def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], K), dtype=float)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y

def label_to_letter(y: np.ndarray) -> list[str]:
    return [letter_map[int(i)] for i in y]
