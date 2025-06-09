import json
import random

# Percorsi dei file
training_file = "training_dataset.jsonl"  # Contiene ID, titolo, abstract, conferenza
results_file = "llama_classification_results.jsonl"  # Contiene ID e relevance_label
train_output_file = "train_dataset.jsonl"  # File per il Training Set (85%)
test_output_file = "test_dataset.jsonl"  # File per il Test Set (15%)

# Carica le label di rilevanza da `llama_classification_results.jsonl`
relevance_labels = {}
with open(results_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 🔹 Ignora righe vuote
        if not line:
            continue
        try:
            data = json.loads(line)
            paper_id = str(data.get("paper_id", "")).strip()
            label = data.get("relevance_label", "").strip()
            if paper_id and label:
                relevance_labels[paper_id] = label
        except json.JSONDecodeError:
            print(f"⚠️ Errore nel parsing della riga: {line}")
            continue  # 🔹 Ignora righe malformate

# Unisce le informazioni nel dataset finale e rimuove i "Non Determinati"
final_data = []
with open(training_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 🔹 Ignora righe vuote
        if not line:
            continue
        try:
            data = json.loads(line)
            paper_id = str(data.get("paper_id", "")).strip()
            relevance_label = relevance_labels.get(paper_id, "Unknown")

            # 🔹 **Escludi i "Non Determinati"**
            if relevance_label == "Non determinato":
                continue

            data["relevance_label"] = relevance_label
            final_data.append(data)
        except json.JSONDecodeError:
            print(f"⚠️ Errore nel parsing della riga: {line}")
            continue  # 🔹 Ignora righe malformate

# Mescola i dati per evitare bias
random.shuffle(final_data)

# Divide il dataset in 85% Training e 15% Test
split_index = int(len(final_data) * 0.85)
train_data = final_data[:split_index]
test_data = final_data[split_index:]

# Scrive i due file separati
with open(train_output_file, "w", encoding="utf-8") as f_out:
    for entry in train_data:
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(test_output_file, "w", encoding="utf-8") as f_out:
    for entry in test_data:
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Report Finale
total_original = len(final_data)
train_size = len(train_data)
test_size = len(test_data)

print("\n=== RISULTATI DIVISIONE DATASET ===")
print(f" Paper totali dopo il filtraggio: {total_original}")
print(f" Training Set (85%): {train_size} paper salvati in {train_output_file}")
print(f" Test Set (15%): {test_size} paper salvati in {test_output_file}")
