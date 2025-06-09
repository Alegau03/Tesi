import os
import json
from collections import defaultdict

# Cartelle di interesse
base_dirs = [
    "data/arxiv.cs.ai_2007-2017",
    "data/arxiv.cs.cl_2007-2017",
    "data/arxiv.cs.lg_2007-2017"
]

# Dizionario per memorizzare le conferenze con gli anni unici
conference_years = defaultdict(set)

# Funzione per estrarre dati dai file JSON
def extract_conference_data(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        conference = data.get("conference")
        date_of_submission = data.get("DATE_OF_SUBMISSION")

        if conference and date_of_submission:
            # Estrarre solo l'anno dalla data
            year = date_of_submission.split("-")[-1]
            conference_years[conference].add(year)
    except Exception as e:
        print(f"Errore nel file {json_path}: {e}")

# Percorrere le cartelle e raccogliere i dati
for base_dir in base_dirs:
    for subset in ["train", "dev", "test"]:
        review_path = os.path.join(base_dir, subset, "reviews")
        if os.path.exists(review_path):
            for file_name in os.listdir(review_path):
                if file_name.endswith(".json"):
                    extract_conference_data(os.path.join(review_path, file_name))

# Convertire il dizionario in una lista di dizionari
output_data = [{"Conferenza": conf, "Anni": sorted(list(years))} for conf, years in conference_years.items()]

# Salvare i dati in un nuovo file JSON
output_file = "Conferenze.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"File '{output_file}' creato con successo!")
