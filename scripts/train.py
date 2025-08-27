import argparse, os, csv
from pathlib import Path
import numpy as np
from letters_nn import NeuralNet, load_csv, one_hot

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--init", choices=["random","zero"], default="random")
    p.add_argument("--out", default="runs/exp")
    args = p.parse_args()

    Xtr, ytr = load_csv(args.train)
    Xva, yva = load_csv(args.valid)

    M = Xtr.shape[1]; K = int(max(ytr.max(), yva.max())) + 1
    Ytr = one_hot(ytr, K); Yva = one_hot(yva, K)

    nn = NeuralNet(M=M, D=args.hidden, K=K, init=args.init, lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    metrics = nn.fit(Xtr, Ytr, Xva, Yva, epochs=args.epochs)

    # save metrics
    with open(Path(args.out)/"metrics.csv","w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_ce","valid_ce"])
        w.writeheader()
        for row in metrics:
            if "valid_ce" not in row: row["valid_ce"] = ""
            w.writerow(row)

    # save model
    nn.save(Path(args.out)/"model.npz")

    # final accuracies
    ytr_pred = nn.predict(Xtr)
    yva_pred = nn.predict(Xva)
    acc_tr = float((ytr_pred == ytr).mean())
    acc_va = float((yva_pred == yva).mean())
    print(f"Done. Train acc={acc_tr:.4f}  Valid acc={acc_va:.4f}")
    with open(Path(args.out)/"acc.txt","w") as f:
        f.write(f"train_acc={acc_tr:.6f}\nvalid_acc={acc_va:.6f}\n")

if __name__ == "__main__":
    main()
