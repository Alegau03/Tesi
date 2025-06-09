#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classifica la rilevanza di un singolo paper e, se non rilevante,
estrae topic con zero-shot e suggerisce una conferenza alternativa,
loggando anche la similarità coseno per debug e escludendo suggerimenti
ridondanti (stessa aggregated root).

Input  : paper.json  (titolo, abstract, conference)
Output : classify_result.json  + stampa su stdout
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────── Config ─────────────────────────────────────────────────────
MODEL_PATH       = "./ModelliFinali/deberta_v3_large/final_model"
CONF_TOPICS_PATH = "ArgomentiConferenze.jsonl"
MERGED_CONFS     = "MergedConferences.json"
PAPER_JSON       = "paper.json"
OUT_JSON         = "classify_result.json"

SOFTMAX_THRESH   = 0.70
SIM_THRESH       = 0.10
ZS_TOP_K         = 5

# ─────────── Fase preliminare: carico il paper ────────────────────────────
paper = json.loads(Path(PAPER_JSON).read_text(encoding="utf-8"))
title     = paper["title"]
abstract  = paper["abstract"]
orig_conf = paper["conference"]  # es. "NIPS 2022"
root_conf = orig_conf.split()[0].lower()

# ─────────── Carico topics per input DeBERTa ─────────────────────────────
conf_topics = {}
with open(CONF_TOPICS_PATH, encoding="utf-8") as f:
    for line in f:
        d   = json.loads(line)
        key = f"{d['Conferenza']} {d['Anno']}"
        conf_topics[key] = ", ".join(d["Argomenti"])
topics_str = conf_topics.get(orig_conf, "")

# ─────────── Fase 1: classificazione con DeBERTa ────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

input_text = f"{title} [SEP] {abstract} [SEP] {orig_conf} [SEP] {topics_str}"
enc = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
).to(device)
with torch.no_grad():
    logits = model(**enc).logits
    probs  = torch.softmax(logits, dim=-1)

pred = "Rilevante" if probs[0,1] >= SOFTMAX_THRESH else "Non rilevante"

# ─────────── Se rilevante: output diretto ───────────────────────────────
suggestion       = "-"
similarity_score = None
extracted_topics = []

if pred == "Non rilevante":
    # ─────────── Fase 2a: zero-shot topic extraction ────────────────
    zs = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    candidate_labels = []
    with open(CONF_TOPICS_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            candidate_labels.extend(d["Argomenti"])
    candidate_labels = list(dict.fromkeys(candidate_labels))

    zs_out = zs(abstract, candidate_labels, multi_label=True)
    extracted_topics = zs_out["labels"][:ZS_TOP_K]

    # ─────────── Fase 2b: TF–IDF retrieval sui topic estratti ──────
    conf_names, conf_corpora = [], []
    with open(MERGED_CONFS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                name = d["Conferenza"]
                text = f"{name} " + " ".join(d["Argomenti"])
                conf_names.append(name)
                conf_corpora.append(text.lower())
            except json.JSONDecodeError:
                parts = line.split('}{')
                for i, part in enumerate(parts):
                    if i > 0: part = '{' + part
                    if i < len(parts)-1: part = part + '}'
                    try:
                        d2 = json.loads(part)
                        name2 = d2["Conferenza"]
                        text2 = f"{name2} " + " ".join(d2["Argomenti"])
                        conf_names.append(name2)
                        conf_corpora.append(text2.lower())
                    except json.JSONDecodeError:
                        continue

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=50000)
    tfidf.fit(conf_corpora)
    conf_vecs = tfidf.transform(conf_corpora)

    blob = " ".join(extracted_topics).lower()
    v    = tfidf.transform([blob])
    sims = cosine_similarity(v, conf_vecs).flatten()

    # seleziono conferenza con sim ≥ soglia, escludendo originale & root
    for idx in sims.argsort()[::-1]:
        cand  = conf_names[idx]
        score = sims[idx]
        if cand.lower() != orig_conf.lower() and cand.lower() != root_conf and score >= SIM_THRESH:
            suggestion       = f"{cand} (similarità: {score:.3f})"
            similarity_score = score
            break
    else:
        suggestion = "Nessuna conferenza trovata"

# ─────────── Output dei risultati ───────────────────────────────────────
result = {
    "paper_id":             paper.get("paper_id", ""),
    "predizione":           pred,
    "conferenza_originale": orig_conf,
    "topics_estratti":      extracted_topics,
    "conferenza_suggerita": suggestion,
    "similarità":           similarity_score
}
Path(OUT_JSON).write_text(json.dumps(result, indent=4), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
