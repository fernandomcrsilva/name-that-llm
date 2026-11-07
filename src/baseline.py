"""Baseline: TF-IDF de caracteres e palavras + regressão logística, validado por prompt.

O teste usa prompts que não aparecem no treino, então a validação separa os folds por prompt_id
(GroupKFold). Validar misturando prompts superestima muito a nota, porque o modelo decora o tema.
uso: python src/baseline.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold


def features():
    return [
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
    ]


def fit_predict(train_text, train_y, test_text):
    vecs = features()
    Xtr = hstack([v.fit_transform(train_text) for v in vecs]).tocsr()
    Xte = hstack([v.transform(test_text) for v in vecs]).tocsr()
    clf = LogisticRegression(C=20, max_iter=3000)
    clf.fit(Xtr, train_y)
    return clf.predict_proba(Xte), clf.classes_


if __name__ == "__main__":
    tr = pd.read_csv("data/train.csv")
    oof = np.zeros(len(tr), dtype=object)
    scores = []
    for k, (a, b) in enumerate(GroupKFold(5).split(tr, groups=tr.prompt_id)):
        proba, classes = fit_predict(tr.text.iloc[a], tr.model.iloc[a], tr.text.iloc[b])
        oof[b] = classes[proba.argmax(1)]
        scores.append(balanced_accuracy_score(tr.model.iloc[b], oof[b]))
        print(f"fold {k}: {scores[-1]:.3f}", flush=True)
    print(f"balanced accuracy (GroupKFold por prompt): {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    if "--confusao" in sys.argv:
        cm = pd.crosstab(tr.model, oof)
        print((cm.div(cm.sum(1), axis=0).round(2)).to_string())
