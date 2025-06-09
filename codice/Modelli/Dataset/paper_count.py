import json

# 📂 Percorso del file JSONL con i risultati
file_path = "llama_classification_results.jsonl"

# 📊 Inizializza i contatori
count_rilevante = 0
count_non_rilevante = 0
count_non_determinato = 0

# 📌 Leggi il file e conta le classificazioni
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # Rimuove spazi vuoti e newline
        if not line:  # Ignora righe vuote
            continue

        try:
            data = json.loads(line)
            label = data.get("relevance_label", "").strip().lower()

            if label == "rilevante":
                count_rilevante += 1
            elif label == "non rilevante":
                count_non_rilevante += 1
            elif label == "non determinato":
                count_non_determinato += 1
        except json.JSONDecodeError:
            print(f"⚠️ Errore nel parsing della riga: {line}")
            continue  # Ignora righe malformate

# 📊 Stampa i risultati
print("\n=== 📊 RISULTATI ===")
print(f"✅ Rilevanti: {count_rilevante}")
print(f"❌ Non Rilevanti: {count_non_rilevante}")
print(f"⚠️ Non Determinati: {count_non_determinato}")
