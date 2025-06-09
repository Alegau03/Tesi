import os
import json
import re
import pandas as pd

# 📂 Percorsi dei dataset
DATA_PATH = "data"  # Cartella con tutte le conferenze
PEERSUM_PATH = "data/peersum"
OUTPUT_FILE = "training_dataset.jsonl"

# 📌 **1. Funzione per estrarre la conferenza da PeerSum (dall'ID)**
def extract_conference_from_id(paper_id):
    match = re.match(r"([a-z]+)_(\d{4})_", paper_id)
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"  # Es. ICLR 2018
    return "Unknown"

# 📌 **2. Funzione per estrarre la conferenza dalle cartelle**
def extract_conference_from_folder(conference_path):
    return os.path.basename(conference_path).replace("_", " ").upper()

# 📌 **3. Funzione per estrarre la conferenza e l'anno dalle cartelle arxiv**
def extract_conference_from_arxiv(data):
    conference_name = data.get("conference", "Unknown").strip()
    date_submission = data.get("DATE_OF_SUBMISSION", "")

    # Estrai l'anno dalla data di submission
    match = re.search(r"(\d{4})", date_submission)
    year = match.group(1) if match else "Unknown"

    return f"{conference_name} {year}".strip()

# 📌 **4. Funzione per processare le review da JSON nelle cartelle delle conferenze**
def process_conference_data():
    processed_data = []

    for conference in os.listdir(DATA_PATH):
        conference_path = os.path.join(DATA_PATH, conference)
        if not os.path.isdir(conference_path):  
            continue  

        # Se è una delle cartelle arxiv, processa diversamente
        is_arxiv = conference in ["arxiv.cs.ai_2007-2017", "arxiv.cs.cl_2007-2017", "arxiv.cs.lg_2007-2017"]

        for split in ["train", "dev", "test"]:
            split_path = os.path.join(conference_path, split, "reviews")
            if not os.path.exists(split_path):
                continue  

            for file in os.listdir(split_path):
                if file.endswith(".json"):
                    file_path = os.path.join(split_path, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    paper_id = data.get("id", "unknown")
                    title = data.get("title", "").strip()
                    abstract = data.get("abstract", "").strip()

                    # Determina la conferenza corretta
                    if is_arxiv:
                        conference_name = extract_conference_from_arxiv(data)
                    else:
                        conference_name = extract_conference_from_folder(conference_path)

                    if title and abstract:
                        processed_data.append({
                            "paper_id": paper_id,
                            "title": title,
                            "abstract": abstract,
                            "conference": conference_name
                        })

    return processed_data

# 📌 **5. Funzione per processare PeerSum (CSV)**
def process_peersum_data():
    processed_data = []
    splits = ["test.csv", "val.csv"]

    for filename in splits:
        csv_path = os.path.join(PEERSUM_PATH, filename)
        if not os.path.exists(csv_path):
            print(f"❌ File {csv_path} non trovato.")
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            paper_id = str(row.get("paper_id", "unknown"))
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()
            conference = extract_conference_from_id(paper_id)

            if title and abstract:
                processed_data.append({
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "conference": conference
                })

    return processed_data

# 📌 **6. Unione e Salvataggio del Dataset**
def save_training_data():
    print("🔄 Processando conferenze...")
    conference_data = process_conference_data()

    print("🔄 Processando PeerSum...")
    peersum_data = process_peersum_data()

    all_data = conference_data + peersum_data
    print(f"✅ Dataset unificato con {len(all_data)} paper.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for entry in all_data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Dataset salvato in {OUTPUT_FILE}")

# 📌 **7. Esegui il programma**
if __name__ == "__main__":
    save_training_data()
