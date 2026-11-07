# Name That LLM: TF-IDF + embeddings do Qwen3-1.7B, sem nenhum truque de grupo de prompt.
# A versão com grupos (validação 0,605) caiu para 0,349 no placar: a estrutura do teste é outra.
# Esta versão só usa o texto de cada passagem. Validação local por prompt: ~0,406.
import glob

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

D = "/kaggle/input/competitions/name-that-llm"
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
classes = sorted(tr.model.unique())
y = tr.model.map({c: i for i, c in enumerate(classes)}).values

# 1) TF-IDF de caracteres e palavras
vecs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]")]
Xtr = hstack([v.fit_transform(tr.text) for v in vecs]).tocsr()
Xte = hstack([v.transform(te.text) for v in vecs]).tocsr()
p_tfidf = LogisticRegression(C=20, max_iter=3000).fit(Xtr, y).predict_proba(Xte)
del Xtr, Xte

# 2) Embeddings do Qwen3-1.7B (camadas 8 e 14 de 28), média dos tokens
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
    A, B = E_tr[:, li], E_te[:, li]
    sc = StandardScaler().fit(A)
    p_emb += LogisticRegression(C=0.02, max_iter=3000).fit(sc.transform(A), y).predict_proba(sc.transform(B))
p_emb /= len(LAYERS)

# 3) Mistura
pred = (0.5 * p_emb + 0.5 * p_tfidf).argmax(1)
sub = pd.DataFrame({"id": te.id, "model": [classes[i] for i in pred]})
sub.to_parquet("submission.parquet", index=False)
print(sub.head(), sub.model.value_counts().head(), sep="\n")
