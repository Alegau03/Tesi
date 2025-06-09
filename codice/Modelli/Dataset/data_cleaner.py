import os
import json
import re
import pandas as pd
from collections import defaultdict

# Percorsi
DATA_PATH = "data"
OUTPUT_FILE = "cleaned_reviews.jsonl"

# 1. Funzione per Pulire il Testo
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # Rimuove spazi multipli
    text = re.sub(r"[^\w\s.,!?;]", "", text)  # Rimuove caratteri speciali
    return text

# 2. Estrarre Review e Accettazione dai JSON nelle cartelle /reviews/
def extract_reviews_from_json():
    processed_data = []
    total_files = 0
    total_reviews = 0
    discarded_papers = 0
    recovered_papers = 0  # Nuovo contatore per i paper senza review ma con "accepted"
    discarded_by_conference = defaultdict(int)  # Conta i paper scartati per conferenza

    for conference in os.listdir(DATA_PATH):
        conference_path = os.path.join(DATA_PATH, conference)
        if not os.path.isdir(conference_path):  
            continue  # Salta i file che non sono directory

        # Scansiona train, dev, test
        for split in ["train", "dev", "test"]:
            split_path = os.path.join(conference_path, split, "reviews")
            if not os.path.exists(split_path):  
                continue  # Salta se la cartella reviews/ non esiste

            for file in os.listdir(split_path):
                if file.endswith(".json"):
                    total_files += 1
                    file_path = os.path.join(split_path, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    paper_id = data.get("id", "unknown")
                    reviews = data.get("reviews", [])

                    all_comments = []
                    recommendations = []

                    # Proviamo a estrarre informazioni utili
                    for review in reviews:
                        comments = clean_text(review.get("comments", ""))
                        if comments:
                            all_comments.append(comments)
                            total_reviews += 1

                        recommendation = review.get("RECOMMENDATION", None)
                        try:
                            if recommendation is not None:
                                recommendation = int(recommendation)
                                recommendations.append(recommendation)
                        except ValueError:
                            continue

                    # Determinare accettazione (se possibile)
                    accepted = data.get("accepted", None)
                    if recommendations:
                        avg_recommendation = sum(recommendations) / len(recommendations)
                        accepted = 1 if avg_recommendation >= 4 else 0

                    # Se il paper non ha review ma ha "accepted", lo recuperiamo!
                    if not all_comments and accepted is not None:
                        recovered_papers += 1
                        all_comments.append(f"Review summary: No review available, but paper was {'Accepted' if accepted == 1 else 'Rejected'}.")

                    # Se il paper non ha informazioni utili, lo scartiamo
                    full_review_text = " ".join(all_comments).strip()
                    if not full_review_text:
                        discarded_papers += 1
                        discarded_by_conference[conference] += 1
                        continue  

                    processed_data.append({
                        "paper_id": paper_id,
                        "review_text": full_review_text,
                        "accepted": accepted  # 1 = Accettato, 0 = Rifiutato, None = Non Determinato
                    })

    print(f"Trovati {total_files} file JSON in data/")
    print(f"Processate {total_reviews} review valide")
    print(f"Recuperati {recovered_papers} paper grazie al campo 'accepted'.")
    print(f"Scartati {discarded_papers} paper perché non avevano né commenti né raccomandazioni.")

    # Stampa delle conferenze che hanno più paper scartati
    if discarded_papers > 0:
        print("\n Paper scartati per conferenza:")
        for conf, count in sorted(discarded_by_conference.items(), key=lambda x: -x[1]):
            print(f"  - {conf}: {count} paper scartati")

    return processed_data

# 3. Unire e Salvare i Dati Puliti
def save_cleaned_data(cleaned_data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset pulito salvato in: {output_file} con {len(cleaned_data)} paper")

# 4. Eseguire la Pulizia e Salvare i Dati
def main():
    print("Pulizia dataset principale...")
    dataset_cleaned = extract_reviews_from_json()
    save_cleaned_data(dataset_cleaned, OUTPUT_FILE)

    print("Pulizia completata!")

if __name__ == "__main__":
    main()
