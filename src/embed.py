"""Extrai embeddings de um LLM congelado (média dos estados ocultos de algumas camadas) para cada texto.

uso: python src/embed.py --model Qwen/Qwen3-1.7B --csv data/train.csv --out runs/emb-qwen3-1.7b-train.npy
Camadas do meio costumam carregar mais estilo; a última carrega mais conteúdo. Salva várias e o
classificador decide.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--csv", default="data/train.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="0.5,0.75,1.0", help="frações da profundidade")
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

texts = pd.read_csv(args.csv).text.tolist()
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dtype = torch.bfloat16 if torch.version.hip or torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = AutoModel.from_pretrained(args.model, dtype=dtype).cuda().eval()
n_layers = model.config.num_hidden_layers
layers = [max(1, round(float(f) * n_layers)) for f in args.layers.split(",")]
order = np.argsort([len(t) for t in texts])  # lotes de tamanho parecido: menos padding
out = np.zeros((len(texts), len(layers), model.config.hidden_size), dtype=np.float32)
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(texts), args.batch):
        idx = order[i:i + args.batch]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        for li, L in enumerate(layers):
            out[idx, li] = ((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy()
np.save(args.out, out)
print(f"{out.shape} salvo em {args.out} | camadas {layers} | {time.time() - t0:.0f}s")
