import json
from collections import defaultdict

# 1. Percorsi dei file
input_file = "ArgomentiConferenze.jsonl"
output_file = "Merged_Conferences.jsonl"

# 2. Leggere e unire le conferenze per nome
conference_dict = defaultdict(set)

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        conference_name = data["Conferenza"]
        topics = set(data["Argomenti"])  # Usa un set per evitare duplicati
        conference_dict[conference_name].update(topics)

# 3. Salvare il nuovo file senza suddivisione per anno
with open(output_file, "w", encoding="utf-8") as f_out:
    for conference, topics in conference_dict.items():
        merged_data = {"Conferenza": conference, "Argomenti": list(topics)}
        f_out.write(json.dumps(merged_data, ensure_ascii=False) + "\n")

print(f"File `{output_file}` creato con tutte le conferenze unite!")
