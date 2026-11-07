"""Quanto os dados extras públicos ajudam? TF-IDF + LinearSVC treinado no treino oficial + pool.

A validação usa só textos oficiais, separados por prompt (GroupKFold). O pool já vem sem os textos
que repetem prompts do treino (ver removal_summary.csv do dataset), então não vaza para a validação.
uso: python src/extra_cv.py --pool 40000 [--sem-proxy] [--peso 0.5]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=40000, help="quantos textos do pool usar (0 = nenhum)")
ap.add_argument("--sem-proxy", action="store_true", help="descarta as linhas com flag_kb_proxy")
ap.add_argument("--peso", type=float, default=0.5, help="peso das linhas extras no treino")
ap.add_argument("--C", type=float, default=0.3)
ap.add_argument("--salvar", default=None, help="grava as probabilidades fora da dobra neste .npy")
args = ap.parse_args()

tr = pd.read_csv("data/train.csv")
classes = sorted(tr.model.unique())
cid = {c: i for i, c in enumerate(classes)}
y = tr.model.map(cid).values

pool = pd.read_parquet("extra/pool/ntl_open_pool.parquet", columns=["text", "model", "flag_kb_proxy"])
if args.sem_proxy:
    pool = pool[~pool.flag_kb_proxy]
# amostra equilibrada: no máximo n/18 por modelo
per = args.pool // len(classes)
pool = pool.sample(frac=1, random_state=0).groupby("model").head(per)
yp = pool.model.map(cid).values
print(f"pool usado: {len(pool)} textos", flush=True)

vecs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=300_000)]
corpus = pd.concat([tr.text, pool.text])
for v in vecs:
    v.fit(corpus)
X = hstack([v.transform(tr.text) for v in vecs]).tocsr()
Xp = hstack([v.transform(pool.text) for v in vecs]).tocsr() if len(pool) else None

oof = np.zeros((len(tr), len(classes)), dtype=np.float32)
for a, b in GroupKFold(5).split(tr, groups=tr.prompt_id):
    Xa, ya, wa = X[a], y[a], np.ones(len(a))
    if Xp is not None:
        Xa, ya, wa = vstack([Xa, Xp]), np.concatenate([ya, yp]), np.concatenate([wa, np.full(len(yp), args.peso)])
    clf = LinearSVC(C=args.C).fit(Xa, ya, sample_weight=wa)
    d = clf.decision_function(X[b])
    e = np.exp(5 * (d - d.max(1, keepdims=True)))
    oof[b] = e / e.sum(1, keepdims=True)
    print(f"  fold: {balanced_accuracy_score(y[b], d.argmax(1)):.3f}", flush=True)
print(f"pool={len(pool)} sem_proxy={args.sem_proxy} peso={args.peso} C={args.C}: "
      f"balanced acc {balanced_accuracy_score(y, oof.argmax(1)):.3f}")
if args.salvar:
    np.save(args.salvar, oof)
