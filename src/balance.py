"""Balanceamento global suave (Sinkhorn): puxa as proporções previstas de cada modelo para perto da meta.

O teste tem quantidades diferentes por modelo, então a meta é uma mistura da proporção prevista com a
uniforme, e o resultado entra só em parte: log P' = (1 - alpha) log P + alpha log Q.
"""
import numpy as np


def sinkhorn(proba, alpha=0.3, mix=0.5, iters=50):
    p = np.clip(proba, 1e-9, 1)
    k = p.shape[1]
    target = mix * p.mean(0) + (1 - mix) / k
    q = p.copy()
    for _ in range(iters):
        q *= (target / q.mean(0))[None, :]
        q /= q.sum(1, keepdims=True)
    out = (1 - alpha) * np.log(p) + alpha * np.log(q)
    return np.exp(out - out.max(1, keepdims=True))


if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score
    tr = pd.read_csv("data/train.csv")
    y = tr.model.map({c: i for i, c in enumerate(sorted(tr.model.unique()))}).values
    o = 0.3 * np.load("runs/emb_pool_C0.05_oof.npy") + 0.7 * np.load("runs/svm100k_oof.npy")
    print(f"sem balancear: {balanced_accuracy_score(y, o.argmax(1)):.3f}")
    rng = np.random.default_rng(0)
    # o treino é equilibrado; para simular o teste, sorteia um subconjunto desequilibrado
    keep = rng.random(len(y)) < rng.uniform(0.3, 1.0, 18)[y]
    for alpha in [0.15, 0.3, 0.5]:
        for mix in [0.0, 0.5]:
            full = balanced_accuracy_score(y, sinkhorn(o, alpha, mix).argmax(1))
            sub = balanced_accuracy_score(y[keep], sinkhorn(o[keep], alpha, mix).argmax(1))
            base = balanced_accuracy_score(y[keep], o[keep].argmax(1))
            print(f"alpha={alpha} mix={mix}: equilibrado {full:.3f} | desequilibrado {sub:.3f} (sem balancear {base:.3f})")
