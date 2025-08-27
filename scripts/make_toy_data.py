# Create a tiny synthetic dataset (not meaningful) for pipeline testing
import numpy as np, csv, os
from pathlib import Path

def gen(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 128))
    y = (X.sum(axis=1) > 64*0.5).astype(int) % 10  # silly label in 0..9
    return X, y

def write_csv(path, X, y):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        for i in range(X.shape[0]):
            w.writerow([int(y[i])] + X[i].tolist())

if __name__ == "__main__":
    Xtr, ytr = gen(300, seed=42)
    Xva, yva = gen(100, seed=43)
    write_csv("data/toyTrain.csv", Xtr, ytr)
    write_csv("data/toyValidation.csv", Xva, yva)
    print("Wrote data/toyTrain.csv and data/toyValidation.csv")
