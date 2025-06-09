from transformers import AutoTokenizer
import json
from transformers import DebertaV2Tokenizer

# Sostituisci con il tuo modello (DeBERTa o Mistral)
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")

# Carica mapping conferenze → argomenti
conf2topics = {}
with open("datasets/A/ArgomentiConferenze.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        key = f"{rec['Conferenza']} {rec['Anno']}"
        conf2topics[key] = " ".join(rec["Argomenti"])

def count_tokens_in_file(path):
    total_tokens = 0
    counts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            key = paper["conference"]                # es. "ICLR 2021"
            text = " ".join([
                paper["title"],
                paper["abstract"],
                key,
                conf2topics.get(key, "")
            ])
            toks = tokenizer(text, add_special_tokens=False)
            n = len(toks["input_ids"])
            total_tokens += n
            counts.append(n)
    return total_tokens, counts

# Conta per train
train_tokens, train_counts = count_tokens_in_file("datasets/A/train_dataset.jsonl")
# Conta per test
test_tokens,  test_counts  = count_tokens_in_file("datasets/A/test_dataset.jsonl")

# Stampa risultati
print(f"TRAIN  → Documenti: {len(train_counts)}, Token totali: {train_tokens}, Media: {train_tokens/len(train_counts):.1f}")
print(f" TEST  → Documenti: {len(test_counts)},  Token totali: {test_tokens}, Media: {test_tokens/len(test_counts):.1f}")
print(f"TOTALE →  Token complessivi: {train_tokens + test_tokens}")
