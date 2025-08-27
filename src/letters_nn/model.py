from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class NeuralNet:
    M: int          # input size
    D: int          # hidden units
    K: int          # classes
    init: str = "random" # "random" or "zero"
    lr: float = 0.1

    def __post_init__(self):
        if self.init not in {"random","zero"}:
            raise ValueError("init must be 'random' or 'zero'")
        if self.init == "random":
            self.alpha = np.random.uniform(-0.1, 0.1, size=(self.D, self.M+1))
            self.beta  = np.random.uniform(-0.1, 0.1, size=(self.K, self.D+1))
            self.alpha[:,0] = 0.0  # bias init 0
            self.beta[:,0]  = 0.0
        else:
            self.alpha = np.zeros((self.D, self.M+1))
            self.beta  = np.zeros((self.K, self.D+1))

    # ---- forward ----
    @staticmethod
    def _sigmoid(a):
        a = np.clip(a, -50, 50)
        return 1.0 / (1.0 + np.exp(-a))

    @staticmethod
    def _softmax(b):
        b = b - b.max(axis=1, keepdims=True)
        e = np.exp(b)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)

    def forward(self, X):
        N = X.shape[0]
        Xb = np.concatenate([np.ones((N,1)), X], axis=1)  # add bias
        a = Xb @ self.alpha.T                    # (N,D)
        z_no_bias = self._sigmoid(a)            # (N,D)
        zb = np.concatenate([np.ones((N,1)), z_no_bias], axis=1)  # (N,D+1)
        b = zb @ self.beta.T                    # (N,K)
        yhat = self._softmax(b)                 # (N,K)
        cache = {"Xb":Xb,"a":a,"z_no_bias":z_no_bias,"zb":zb,"b":b,"yhat":yhat}
        return yhat, cache

    @staticmethod
    def cross_entropy(yhat, Y):
        # Y: one-hot (N,K)
        eps = 1e-12
        return -np.mean(np.sum(Y * np.log(yhat + eps), axis=1))

    # ---- backward (softmax + CE combo) ----
    def backward(self, cache, Y):
        # cache from forward; Y is one-hot
        N = Y.shape[0]
        Xb = cache["Xb"]
        zb = cache["zb"]
        z_no_bias = cache["z_no_bias"]
        yhat = cache["yhat"]

        # dL/db = (yhat - Y)
        G_b = (yhat - Y) / N                    # (N,K)

        # grad beta: g_beta = G_b^T @ zb
        g_beta = G_b.T @ zb                      # (K, D+1)

        # backprop to z (exclude bias column when passing through sigmoid)
        G_z = G_b @ self.beta                    # (N, D+1)
        G_z_no_bias = G_z[:,1:]                  # (N,D)

        # sigmoid grad: z' = z*(1-z)
        G_a = G_z_no_bias * (z_no_bias * (1.0 - z_no_bias))  # (N,D)

        # grad alpha: g_alpha = G_a^T @ Xb
        g_alpha = G_a.T @ Xb                     # (D, M+1)

        return g_alpha, g_beta

    # ---- SGD step ----
    def sgd_step(self, X, Y):
        yhat, cache = self.forward(X)
        g_alpha, g_beta = self.backward(cache, Y)
        self.alpha -= self.lr * g_alpha
        self.beta  -= self.lr * g_beta
        return self.cross_entropy(yhat, Y)

    def fit(self, Xtr, Ytr, Xva=None, Yva=None, epochs=10):
        metrics = []
        for e in range(1, epochs+1):
            loss_tr = self.sgd_step(Xtr, Ytr)
            row = {"epoch": e, "train_ce": float(loss_tr)}
            if Xva is not None and Yva is not None:
                yhat_va, _ = self.forward(Xva)
                row["valid_ce"] = float(self.cross_entropy(yhat_va, Yva))
            metrics.append(row)
        return metrics

    def predict(self, X):
        yhat, _ = self.forward(X)
        return np.argmax(yhat, axis=1)

    # ---- save/load ----
    def save(self, path):
        np.savez(path, alpha=self.alpha, beta=self.beta, M=self.M, D=self.D, K=self.K, lr=self.lr)
    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        nn = cls(int(d["M"]), int(d["D"]), int(d["K"]), init="zero", lr=float(d["lr"]))
        nn.alpha = d["alpha"]; nn.beta = d["beta"]
        return nn
