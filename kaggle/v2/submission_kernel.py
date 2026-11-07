# Name That LLM: TF-IDF + embeddings do Qwen3-1.7B, ambos centrados por grupo de prompt,
# com atribuição húngara "cada modelo uma vez por prompt". Validação local por prompt: ~0,605.
import glob
import math

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

D = "/kaggle/input/competitions/name-that-llm"
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
classes = sorted(tr.model.unique())
y = tr.model.map({c: i for i, c in enumerate(classes)}).values


def prompt_groups(texts, threshold=0.9):
    # Reconstrói os grupos de prompt do teste por similaridade de conteúdo
    if len(texts) < 20:
        return np.arange(len(texts))
    X = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, ngram_range=(1, 2)).fit_transform(texts)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                   linkage="average").fit_predict(X.toarray())


def center(X, groups):
    # Tira o "tema médio" de cada grupo de prompt, para sobrar o estilo
    X = X.copy()
    for g in np.unique(groups):
        m = groups == g
        X[m] -= X[m].mean(0)
    return X


def assign(proba, groups):
    k = proba.shape[1]
    pred = proba.argmax(1)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        cap = math.ceil(len(idx) / k)
        r, c = linear_sum_assignment(-np.log(np.repeat(proba[idx], cap, axis=1) + 1e-9))
        pred[idx[r]] = c // cap
    return pred


g_tr, g_te = tr.prompt_id.values, prompt_groups(te.text.tolist())

# 1) TF-IDF de caracteres e palavras, centrado no grupo
vecs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=40000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=20000)]
Xtr = center(hstack([v.fit_transform(tr.text) for v in vecs]).toarray().astype(np.float32), g_tr)
Xte = center(hstack([v.transform(te.text) for v in vecs]).toarray().astype(np.float32), g_te)
p_tfidf = LogisticRegression(C=20, max_iter=3000).fit(Xtr, y).predict_proba(Xte)
del Xtr, Xte

# 2) Embeddings do Qwen3-1.7B (camadas 8 e 14 de 28), média dos tokens, centrados no grupo
path = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
path = [p for p in path if "1.7b" in p.lower() or "1_7b" in p.lower()][0].rsplit("/", 1)[0]
tok = AutoTokenizer.from_pretrained(path)
tok.padding_side = "right"
model = AutoModel.from_pretrained(path, dtype=torch.float16).cuda().eval()
LAYERS = [8, 14]


def embed(texts):
    out = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            idx = order[i:i + 16]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            m = enc["attention_mask"].unsqueeze(-1).float()
            for li, L in enumerate(LAYERS):
                out[idx, li] = ((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy()
    return out


E_tr, E_te = embed(tr.text.tolist()), embed(te.text.tolist())
p_emb = np.zeros_like(p_tfidf)
for li in range(len(LAYERS)):
    A, B = center(E_tr[:, li], g_tr), center(E_te[:, li], g_te)
    sc = StandardScaler().fit(A)
    p_emb += LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(A), y).predict_proba(sc.transform(B))
p_emb /= len(LAYERS)

# 3) Mistura e atribuição por grupo
pred = assign(0.5 * p_emb + 0.5 * p_tfidf, g_te)
sub = pd.DataFrame({"id": te.id, "model": [classes[i] for i in pred]})
sub.to_parquet("submission.parquet", index=False)
print(sub.head(), sub.model.value_counts().head(), sep="\n")
