#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline 2-step per 100 paper con logging della similarità coseno,
escludendo suggerimenti ridondanti quando la venue aggregata coincide
con la radice della conferenza originale (e.g., "NIPS" vs "NIPS 2022").

1) DeBERTa fine-tuned classifica Rilevante vs Non rilevante (softmax ≥ 0.70)
2) Se Non rilevante, Zero-Shot topic extraction + TF-IDF retrieval
   estrae topic dall’abstract e suggerisce una conferenza (esclude l’originale
   e le aggregated venue corrispondenti) includendo nel JSON la similarità.

Output:
  • 100_random_papers_2step.json
  • suggestions_2step.txt
"""

import json
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────── Config ─────────────────────────────────────────────────────
MODEL_PATH        = "./ModelliFinali/deberta_v3_large/final_model"
MERGED_CONF_PATH  = "FileImportanti/MergedConferences.json"
CONF_TOPICS_PATH  = "FileImportanti/ArgomentiConferenze.jsonl"
TEST_FILE         = "FileImportanti/test_dataset.jsonl"
OUT_JSON          = "100_random_papers_2step.json"
OUT_TXT           = "suggestions_2step.txt"

N_SAMPLES         = 100
SEED              = 42
SOFTMAX_THRESH    = 0.70    # soglia per DeBERTa
SIM_THRESH        = 0.10    # soglia per TF-IDF retrieval
ZS_TOP_K          = 5       # numero di topic da estrarre

random.seed(SEED)

# ─────────── Carica test-set ─────────────────────────────────────────────
with open(TEST_FILE, encoding="utf-8") as f:
    all_papers = [json.loads(line) for line in f]

# ─────────── Carica conferenze note (per retrieval) ──────────────────────
conf_names, conf_corpora = [], []
with open(MERGED_CONF_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            conf_names.append(d["Conferenza"])
            text = f"{d['Conferenza']} " + " ".join(d["Argomenti"])
            conf_corpora.append(text.lower())
        except json.JSONDecodeError:
            # split in caso di concatenazioni senza newline
            parts = line.split('}{')
            for i, part in enumerate(parts):
                if i > 0:
                    part = '{' + part
                if i < len(parts) - 1:
                    part = part + '}'
                try:
                    d2 = json.loads(part)
                    conf_names.append(d2["Conferenza"])
                    text2 = f"{d2['Conferenza']} " + " ".join(d2["Argomenti"])
                    conf_corpora.append(text2.lower())
                except json.JSONDecodeError:
                    continue

# ─────────── Prepara TF-IDF su (conferenze ∪ abstracts) ─────────────────
paper_corpus = [f"{p['title']} {p['abstract']}".lower() for p in all_papers]
tfidf_vec    = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=50000)
tfidf_vec.fit(conf_corpora + paper_corpus)
conf_vectors = tfidf_vec.transform(conf_corpora)

def best_conference(blob, orig_conf=None, thresh=SIM_THRESH):
    """
    Restituisce (conference_name, similarity) con sim TF-IDF massima (≥ thresh),
    escludendo orig_conf e la sua aggregated root (e.g., "NIPS").
    """
    v    = tfidf_vec.transform([blob.lower()])
    sims = cosine_similarity(v, conf_vectors).flatten()
    # radice della conferenza originale (prima parola)
    root = orig_conf.split()[0].lower() if orig_conf else ""
    for idx in sims.argsort()[::-1]:
        cand  = conf_names[idx]
        score = sims[idx]
        # escludi stessa full name e aggregated root
        if cand.lower() != orig_conf.lower() and cand.lower() != root and score >= thresh:
            return cand, score
    return "Nessuna conferenza trovata", 0.0

# ─────────── Carica topic-by-conference+anno (per DeBERTa input) ─────────
conf_topics = {}
with open(CONF_TOPICS_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        key = f"{d['Conferenza']} {d['Anno']}"
        conf_topics[key] = ", ".join(d["Argomenti"])

# ─────────── Inizializza DeBERTa fine-tuned ─────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# ─────────── Inizializza Zero-Shot Classification pipeline ─────────────
zs_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

# ─────────── Costruisci lista candidate_labels dai topic JSONL ─────────
candidate_labels = []
with open(CONF_TOPICS_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        candidate_labels.extend(d["Argomenti"])
candidate_labels = list(dict.fromkeys(candidate_labels))

# ─────────── Campiona e processa ────────────────────────────────────────
sample = random.sample(all_papers, N_SAMPLES)
results = []
suggestions = []

for rec in sample:
    paper_id = rec["paper_id"]
    title    = rec["title"]
    abstract = rec["abstract"]
    conf_key = rec["conference"]  # include anno

    # STEP 1: DeBERTa relevance
    topics_str = conf_topics.get(conf_key, "")
    model_text = f"{title} [SEP] {abstract} [SEP] {conf_key} [SEP] {topics_str}"
    enc = tokenizer(
        model_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    pred = "Rilevante" if probs[0,1] >= SOFTMAX_THRESH else "Non rilevante"

    # STEP 2: suggestion via Zero-Shot + TF-IDF
    suggestion = "-"
    sim_score  = None
    extracted  = []

    if pred == "Non rilevante":
        # 2a: estrai top-k topic
        zs_out     = zs_pipeline(abstract, candidate_labels, multi_label=True)
        top_topics = zs_out["labels"][:ZS_TOP_K]
        extracted  = top_topics

        # 2b: retrieval su blob di topics
        blob        = " ".join(top_topics)
        cand, score = best_conference(blob, orig_conf=conf_key)
        suggestion  = f"{cand} (similarità: {score:.3f})"
        sim_score   = score

        suggestions.append(f"{paper_id}, {conf_key}, {suggestion}")

    # raccogli risultati
    results.append({
        "paper_id":             paper_id,
        "titolo":               title[:120],
        "predizione":           pred,
        "conferenza_originale": conf_key,
        "topics_estratti":      extracted,
        "conferenza_suggerita": suggestion,
        "similarità":           sim_score
    })

# ─────────── Salva output ───────────────────────────────────────────────
Path(OUT_JSON).write_text(json.dumps(results, indent=4), encoding="utf-8")
Path(OUT_TXT).write_text("\n".join(suggestions),                   encoding="utf-8")

print("100_random_papers_2step.json creato.")
print("suggestions_2step.txt creato.")
