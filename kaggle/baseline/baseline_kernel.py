# Name That LLM: baseline TF-IDF (caracteres + palavras) com regressão logística.
# Validação local por prompt (GroupKFold): balanced accuracy ~0,34.
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

D = "/kaggle/input/competitions/name-that-llm"
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
vecs = [
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000),
    TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                    token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
]
Xtr = hstack([v.fit_transform(tr.text) for v in vecs]).tocsr()
Xte = hstack([v.transform(te.text) for v in vecs]).tocsr()
clf = LogisticRegression(C=20, max_iter=3000).fit(Xtr, tr.model)
pd.DataFrame({"id": te.id, "model": clf.predict(Xte)}).to_parquet("submission.parquet", index=False)
print(pd.read_parquet("submission.parquet").head())
