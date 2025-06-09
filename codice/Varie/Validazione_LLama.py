import json
import random
import pandas as pd
import os

# === Percorsi dei file ===
REVIEWS_FILES      = ["cleaned_reviews.jsonl", "cleaned_peersum.jsonl"]
LLAMA_RESULTS_FILE = "llama_classification_results.jsonl"
OUTPUT_EXCEL       = "validazione_10_paper.xlsx"
DESIRED_PER_CAT    = 10
TOTAL_DESIRED      = 20

# === 1. Carica etichette da LLaMA ===
def load_llama_labels():
    llama = {}
    with open(LLAMA_RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            lbl = d.get("relevance_label")
            if lbl in ["Rilevante", "Non rilevante"]:
                llama[d["paper_id"]] = lbl
    return llama

# === 2. Carica review valide ===
def load_reviews():
    reviews = {}
    for path in REVIEWS_FILES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid  = d.get("paper_id")
                text = d.get("review_text", "").strip()
                # Escludi placeholder “no review available”
                if not pid or not text or "no review available" in text.lower():
                    continue
                # Solo paper rifiutati
                if d.get("accepted") in [False, 0]:
                    reviews[pid] = text
    return reviews

# === MAIN ===
llama_labels = load_llama_labels()
reviews      = load_reviews()

# Separa gli ID per categoria
relevant_ids = [pid for pid, lbl in llama_labels.items() if lbl == "Rilevante"     and pid in reviews]
nonrel_ids   = [pid for pid, lbl in llama_labels.items() if lbl == "Non rilevante" and pid in reviews]

print(f"Trovati {len(relevant_ids)} rilevanti e {len(nonrel_ids)} non rilevanti con review valida.")

# Imposta semi per riproducibilità
random.seed(42)

# Campiona fino a DESIRED_PER_CAT per categoria
sample_rel    = random.sample(relevant_ids, min(DESIRED_PER_CAT, len(relevant_ids)))
sample_nonrel = random.sample(nonrel_ids,   min(DESIRED_PER_CAT, len(nonrel_ids)))

# Unisci e calcola quanti ne mancano
selected = set(sample_rel + sample_nonrel)
remaining_needed = TOTAL_DESIRED - len(selected)

if remaining_needed > 0:
    # Pool di tutti gli ID rimanenti
    pool = [pid for pid in (relevant_ids + nonrel_ids) if pid not in selected]
    if len(pool) < remaining_needed:
        raise RuntimeError(f"Non ci sono abbastanza paper (mancano {remaining_needed}).")
    extra = random.sample(pool, remaining_needed)
    # Decidi se marcati come “rilevanti” o “non rilevanti” in base al label originale
    sample_rel    += [pid for pid in extra if llama_labels[pid] == "Rilevante"]
    sample_nonrel += [pid for pid in extra if llama_labels[pid] == "Non rilevante"]
    selected.update(extra)

# Ora “selected” contiene esattamente TOTAL_DESIRED paper
final_ids = list(selected)
print(f"In totale estratti {len(final_ids)} paper (desiderati: {TOTAL_DESIRED}).")

# === Costruisci DataFrame e salva in Excel ===
rows = []
for pid in final_ids:
    rows.append({
        "PAPER ID": pid,
        "REVIEW": reviews[pid],
        "RISPOSTA DI LLAMA": 1 if llama_labels[pid] == "Rilevante" else 0,
        "MIA RISPOSTA": "",
        "RISPOSTA PROFESSORE": ""
    })

df = pd.DataFrame(rows)
# ordina per risposta (rilevanti prima)
df["__order"] = df["RISPOSTA DI LLAMA"].map({1: 0, 0: 1})
df = df.sort_values("__order").drop(columns="__order")

df.to_excel(OUTPUT_EXCEL, index=False)
print(f"✅ Creato file {OUTPUT_EXCEL} con {len(df)} paper.")
