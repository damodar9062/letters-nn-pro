import argparse
from pathlib import Path
import numpy as np
from letters_nn import NeuralNet, load_csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    nn = NeuralNet.load(args.model)
    X, y = load_csv(args.data)
    y_pred = nn.predict(X)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(map(str, y_pred.tolist())), encoding="utf-8")
    if y is not None:
        acc = float((y_pred == y).mean())
        print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
