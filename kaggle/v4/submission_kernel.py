# Name That LLM v4: TF-IDF + LinearSVC e embeddings do Qwen3-1.7B, os dois treinados com o treino oficial
# mais textos públicos do Name That LLM Open Pool (crtproduction, CC BY / CC BY-SA), e balanceamento
# global suave (Sinkhorn). Validação local por prompt: 0,511 (baseline 0,342).
import glob

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

D = "/kaggle/input/competitions/name-that-llm"
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
classes = sorted(tr.model.unique())
cid = {c: i for i, c in enumerate(classes)}
y = tr.model.map(cid).values
pool_all = pd.read_parquet(glob.glob("/kaggle/input/**/ntl_open_pool.parquet", recursive=True)[0],
                           columns=["text", "model"])
pool_all = pool_all.sample(frac=1, random_state=0)


def balanced_sample(n):
    return pool_all.groupby("model").head(n // len(classes))


# 1) TF-IDF + LinearSVC com 100 mil textos do pool (peso 0,5)
pool = balanced_sample(100_000)
yp = pool.model.map(cid).values
vecs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=300_000)]
for v in vecs:
    v.fit(pd.concat([tr.text, pool.text]))
X = vstack([hstack([v.transform(tr.text) for v in vecs]), hstack([v.transform(pool.text) for v in vecs])]).tocsr()
w = np.concatenate([np.ones(len(tr)), np.full(len(pool), 0.5)])
svm = LinearSVC(C=0.3).fit(X, np.concatenate([y, yp]), sample_weight=w)
d = svm.decision_function(hstack([v.transform(te.text) for v in vecs]).tocsr())
p_svm = np.exp(5 * (d - d.max(1, keepdims=True)))
p_svm /= p_svm.sum(1, keepdims=True)
del X, svm
print("SVM pronto", flush=True)

# 2) Embeddings do Qwen3-1.7B (camadas 8 e 14), regressão logística com 40 mil textos do pool (peso 0,5)
pool = balanced_sample(40_000)
yp = pool.model.map(cid).values
path = [p for p in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
        if "1.7b" in p.lower()][0].rsplit("/", 1)[0]
tok = AutoTokenizer.from_pretrained(path)
tok.padding_side = "right"
model = AutoModel.from_pretrained(path, dtype=torch.float16).cuda().eval()
LAYERS = [8, 14]


def embed(texts):
    out = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            idx = order[i:i + 32]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            m = enc["attention_mask"].unsqueeze(-1).float()
            for li, L in enumerate(LAYERS):
                out[idx, li] = ((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy()
    return out


E_tr, E_pool, E_te = embed(tr.text.tolist()), embed(pool.text.tolist()), embed(te.text.tolist())
print("embeddings prontos", flush=True)
p_emb = np.zeros_like(p_svm)
w = np.concatenate([np.ones(len(tr)), np.full(len(pool), 0.5)])
for li in range(len(LAYERS)):
    A = np.concatenate([E_tr[:, li], E_pool[:, li]])
    sc = StandardScaler().fit(A)
    lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(A), np.concatenate([y, yp]), sample_weight=w)
    p_emb += lr.predict_proba(sc.transform(E_te[:, li])) / len(LAYERS)


# 3) Mistura e balanceamento global suave
def sinkhorn(proba, alpha=0.3, mix=0.5, iters=50):
    p = np.clip(proba, 1e-9, 1)
    target = mix * p.mean(0) + (1 - mix) / p.shape[1]
    q = p.copy()
    for _ in range(iters):
        q *= (target / q.mean(0))[None, :]
        q /= q.sum(1, keepdims=True)
    return (1 - alpha) * np.log(p) + alpha * np.log(q)


pred = sinkhorn(0.3 * p_emb + 0.7 * p_svm).argmax(1)
sub = pd.DataFrame({"id": te.id, "model": [classes[i] for i in pred]})
sub.to_parquet("submission.parquet", index=False)
print(sub.model.value_counts().head(), flush=True)
