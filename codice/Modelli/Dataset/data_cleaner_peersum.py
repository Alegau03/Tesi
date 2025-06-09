import os
import json
import pandas as pd
import re
import torch
from transformers import AutoTokenizer

# Percorsi
DATA_PATH = "data"
PEERSUM_PATH = "peersum"
OUTPUT_FILE = "cleaned_dataset2.jsonl"
PEERSUM_FILE = "cleaned_peersum2.jsonl"



# 1. Funzione per Pulire il Testo
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # Rimuove spazi multipli
    text = re.sub(r"[^\w\s.,!?;]", "", text)  # Rimuove caratteri speciali
    return text

# 2. Estrarre le Review dai JSON
def extract_reviews_from_json(conference_dir):
    processed_data = []
    
    for file in os.listdir(conference_dir):
        if file.endswith(".json"):
            file_path = os.path.join(conference_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            paper_id = data.get("id", "unknown")
            title = clean_text(data.get("title", ""))
            abstract = clean_text(data.get("abstract", ""))
            reviews = data.get("reviews", [])

            all_comments = []
            relevance_labels = []

            for review in reviews:
                # Estrai i commenti
                comments = clean_text(review.get("comments", ""))
                if comments:
                    all_comments.append(comments)

                # Determina la rilevanza basata su RECOMMENDATION
                recommendation = review.get("RECOMMENDATION", "0")
                try:
                    recommendation = int(recommendation)
                    if recommendation >= 4:
                        relevance_labels.append("Rilevante")
                    else:
                        relevance_labels.append("Non rilevante")
                except ValueError:
                    pass  # Se non è un numero, ignora

            # Unisce tutti i commenti in un unico testo
            full_review_text = " ".join(all_comments).strip()

            # Determina la label finale
            if not full_review_text:
                final_label = "Non determinato"
            elif "Rilevante" in relevance_labels:
                final_label = "Rilevante"
            elif "Non rilevante" in relevance_labels:
                final_label = "Non rilevante"
            else:
                final_label = "Non determinato"

            # Tokenizza il testo se richiesto
            if USE_TOKENIZATION:
                full_review_text = TOKENIZER.decode(
                    TOKENIZER.encode(full_review_text, truncation=True, max_length=512)
                )

            processed_data.append({
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "review_text": full_review_text,
                "relevance_label": final_label
            })

    return processed_data

# 3. Estrarre le Review da Peersum (CSV)
def extract_reviews_from_peersum():
    processed_data = []
    
    # Mappa dei file CSV
    splits = {"train": "train.csv", "test": "test.csv", "dev": "val.csv"}

    for split, filename in splits.items():
        csv_path = os.path.join(PEERSUM_PATH, filename)
        if not os.path.exists(csv_path):
            print(f"❌ File {csv_path} non trovato.")
            continue

        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            abstract = clean_text(row.get("abstract", ""))
            reviews = row.get("reviews", "[]")  # Lista di recensioni
            meta_review = clean_text(row.get("meta_review", ""))  # Recensione principale
            decision = str(row.get("review_ratings", "")).lower()  # Decisione finale

            # Converti le review da stringa JSON a lista
            try:
                reviews_list = json.loads(reviews)
                reviews_text = " ".join([clean_text(review) for review in reviews_list])
            except json.JSONDecodeError:
                reviews_text = ""

            full_review = f"{reviews_text} {meta_review}".strip()

            # Converti decisione in etichetta binaria
            if "8" in decision or "9" in decision:
                label = "Rilevante"
            elif "3" in decision or "4" in decision:
                label = "Non rilevante"
            else:
                label = "Non determinato"

            # Tokenizza il testo se necessario
            if USE_TOKENIZATION:
                full_review = TOKENIZER.decode(
                    TOKENIZER.encode(full_review, truncation=True, max_length=512)
                )

            processed_data.append({
                "abstract": abstract,
                "review_text": full_review,
                "relevance_label": label
            })

    return processed_data

# 4. Unire e Salvare i Dati Puliti
def save_cleaned_data(cleaned_data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset pulito salvato in: {output_file}")

# 5. Eseguire la Pulizia e Salvare i Dati
def main():
    # Pulisci il dataset principale
    print("Pulizia dataset principale...")
    dataset_cleaned = extract_reviews_from_json(DATA_PATH)
    save_cleaned_data(dataset_cleaned, OUTPUT_FILE)

    # Pulisci il dataset Peersum
    print("Pulizia dataset Peersum...")
    peersum_cleaned = extract_reviews_from_peersum()
    save_cleaned_data(peersum_cleaned, PEERSUM_FILE)

    print("Pulizia completata!")

if __name__ == "__main__":
    main()