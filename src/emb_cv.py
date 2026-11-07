"""Embeddings do Qwen3-1.7B (camadas 8 e 14) + regressão logística, com e sem 40 mil textos do pool.

Usa os embeddings já extraídos por src/embed.py (treino: camada 8 do arquivo "shallow" e 14 do
principal; pool: camadas 8 e 14). No fim, mistura com o SVM de extra_cv.py e aplica o Sinkhorn.
Validação por prompt (GroupKFold), igual ao resto do projeto.
uso: python src/extra_cv.py --pool 100000 --salvar runs/svm100k_oof.npy && python src/emb_cv.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from balance import sinkhorn

tr = pd.read_csv("data/train.csv")
classes = sorted(tr.model.unique())
cid = {c: i for i, c in enumerate(classes)}
y = tr.model.map(cid).values
pool = pd.read_csv("extra/pool_sample40k.csv")
yp = pool.model.map(cid).values
E_tr = np.stack([np.load("runs/emb-qwen3-1.7b-shallow-train.npy")[:, 1],
                 np.load("runs/emb-qwen3-1.7b-train.npy")[:, 0]], axis=1)
E_pool = np.load("runs/emb-q17-pool40k.npy")


def cv(com_pool):
    oof = np.zeros((len(tr), len(classes)))
    for a, b in GroupKFold(5).split(tr, groups=tr.prompt_id):
        for li in range(E_tr.shape[1]):
            A, ya, w = E_tr[a, li], y[a], np.ones(len(a))
            if com_pool:
                A = np.concatenate([A, E_pool[:, li]])
                ya = np.concatenate([ya, yp])
                w = np.concatenate([w, np.full(len(yp), 0.5)])
            sc = StandardScaler().fit(A)
            lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(A), ya, sample_weight=w)
            oof[b] += lr.predict_proba(sc.transform(E_tr[b, li])) / E_tr.shape[1]
    return oof


ba = lambda p: balanced_accuracy_score(y, p.argmax(1))
print(f"embeddings sem pool: {ba(cv(False)):.3f}", flush=True)
emb = cv(True)
print(f"embeddings com pool: {ba(emb):.3f}", flush=True)
mix = 0.3 * emb + 0.7 * np.load("runs/svm100k_oof.npy")
print(f"mistura 0,3 emb + 0,7 SVM: {ba(mix):.4f} | com Sinkhorn: {ba(sinkhorn(mix)):.4f}")
