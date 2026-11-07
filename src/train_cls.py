"""Classificador de autoria: LLM pequeno (Qwen3) com cabeça de classificação e LoRA, validado por prompt.

uso: python src/train_cls.py --model Qwen/Qwen3-0.6B --folds 0        # um fold, para medir
     python src/train_cls.py --model Qwen/Qwen3-0.6B --full            # treino final em todo o treino
Grava as probabilidades fora da dobra em runs/<nome>/oof.npy, para depois combinar com o TF-IDF.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--full", action="store_true")
ap.add_argument("--epochs", type=float, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--maxlen", type=int, default=384)
ap.add_argument("--name", default=None)
args = ap.parse_args()

tr = pd.read_csv("data/train.csv")
classes = sorted(tr.model.unique())
y = tr.model.map({c: i for i, c in enumerate(classes)}).values
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
enc = tok(list(tr.text), truncation=True, max_length=args.maxlen)["input_ids"]
name = args.name or args.model.split("/")[-1].lower()
out = Path("runs") / name
out.mkdir(parents=True, exist_ok=True)


def batches(idx, size, shuffle, rng=None):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        chunk = idx[i:i + size]
        n = -(-max(len(enc[j]) for j in chunk) // 32) * 32
        ids = torch.full((len(chunk), n), tok.pad_token_id)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, j in enumerate(chunk):
            ids[k, :len(enc[j])] = torch.tensor(enc[j])
            mask[k, :len(enc[j])] = 1
        yield ids.cuda(), mask.cuda(), torch.tensor(y[chunk]).cuda(), chunk


def train(train_idx, eval_idx=None):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes), dtype=torch.bfloat16).cuda()
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # a cabeça nova e o LoRA treinam em fp32; o resto fica em bf16
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train_idx) / args.batch * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    step, micro, t0 = 0, 0, time.time()
    model.train()
    while step < steps:
        for ids, mask, yy, _ in batches(train_idx, args.batch, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            (torch.nn.functional.cross_entropy(logits.float(), yy) / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"  passo {step}/{steps} | {time.time() - t0:.0f}s", flush=True)
            if step >= steps:
                break
    proba = None
    if eval_idx is not None:
        model.eval()
        proba = np.zeros((len(eval_idx), len(classes)), dtype=np.float32)
        pos = {j: k for k, j in enumerate(eval_idx)}
        with torch.no_grad():
            for ids, mask, _, chunk in batches(eval_idx, 16, False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1).cpu().numpy()
                for k, j in enumerate(chunk):
                    proba[pos[j]] = p[k]
    return model, proba


if args.full:
    model, _ = train(np.arange(len(tr)))
    model.save_pretrained(out / "full")
    tok.save_pretrained(out / "full")
    (out / "full" / "classes.txt").write_text("\n".join(classes) + "\n")
    print(f"modelo final salvo em {out / 'full'}")
else:
    oof_path = out / "oof.npy"
    oof = np.load(oof_path) if oof_path.exists() else np.full((len(tr), len(classes)), np.nan, dtype=np.float32)
    splits = list(GroupKFold(5).split(tr, groups=tr.prompt_id))
    for f in map(int, args.folds.split(",")):
        a, b = splits[f]
        t0 = time.time()
        _, proba = train(a, b)
        oof[b] = proba
        np.save(oof_path, oof)
        print(f"fold {f}: balanced acc {balanced_accuracy_score(y[b], proba.argmax(1)):.3f} | {time.time() - t0:.0f}s", flush=True)
    done = ~np.isnan(oof[:, 0])
    print(f"OOF ({done.sum()} textos): {balanced_accuracy_score(y[done], oof[done].argmax(1)):.3f}")
