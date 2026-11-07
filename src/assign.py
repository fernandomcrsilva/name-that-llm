"""Pós-processamento: reconstrói os grupos de prompt por tema e impõe "cada modelo uma vez por prompt".

No treino, cada prompt foi respondido uma única vez por cada um dos 18 modelos. O teste não traz o
prompt_id, então os textos são agrupados por similaridade de conteúdo (TF-IDF de palavras, sem stop words).
Dentro de cada grupo, a atribuição texto→modelo maximiza a soma de log-probabilidades (algoritmo húngaro).
Se um grupo tiver mais de 18 textos (dois prompts juntos por engano), cada modelo pode aparecer
ceil(n/18) vezes, para a restrição não forçar erros.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


def prompt_groups(texts, threshold=0.9):
    X = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, ngram_range=(1, 2)).fit_transform(texts)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                   linkage="average").fit_predict(X.toarray())


def assign(proba, groups):
    k = proba.shape[1]
    pred = proba.argmax(1)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        cap = math.ceil(len(idx) / k)
        cost = -np.log(np.repeat(proba[idx], cap, axis=1) + 1e-9)  # coluna j*cap+r = modelo j, vaga r
        r, c = linear_sum_assignment(cost)
        pred[idx[r]] = c // cap
    return pred


if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    tr = pd.read_csv("data/train.csv")
    classes = sorted(tr.model.unique())
    y = tr.model.map({c: i for i, c in enumerate(classes)}).values
    oof = np.load("runs/tfidf_oof.npy")
    pred = np.zeros(len(tr), dtype=int)
    for a, b in GroupKFold(5).split(tr, groups=tr.prompt_id):
        pred[b] = assign(oof[b], prompt_groups(tr.text.iloc[b].tolist()))
    print(f"argmax {balanced_accuracy_score(y, oof.argmax(1)):.3f} -> grupos reconstruídos + húngaro {balanced_accuracy_score(y, pred):.3f}")
    assert assign(np.eye(3)[[0, 0, 1]] * 0.9 + 0.05, np.zeros(3, int)).tolist().count(0) == 1
