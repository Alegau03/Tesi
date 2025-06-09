import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
from huggingface_hub import login
login()
# Percorsi dei file
train_file = "train_dataset.jsonl"
test_file = "test_dataset.jsonl"
conference_topics_file = "ArgomentiConferenze.jsonl"
train_output = "FileTokenizzati/tokenizzatiMistral/train_tokenized_mistral.pt"
test_output = "FileTokenizzati/tokenizzatiMistral/test_tokenized_mistral.pt"

# Carica il tokenizer di Mistral 7B
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Fix per il padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Imposta EOS come token di padding
    print("✅ Token di padding impostato su EOS Token")

# Carica gli argomenti delle conferenze
conference_topics = {}
with open(conference_topics_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            conf_key = f"{data['Conferenza']} {data['Anno']}"  # Es: "ICLR 2018"
            conference_topics[conf_key] = ", ".join(data["Argomenti"])
            print(f"🔍 {conf_key}: {conference_topics[conf_key]}")
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing di una riga: {e}")

# Funzione per tokenizzare i dati e includere gli argomenti della conferenza
def tokenize_dataset(input_file, output_file):
    paper_ids = []
    input_ids = []
    attention_masks = []
    labels = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                paper_id = data["paper_id"]
                title = data["title"]
                abstract = data["abstract"]
                conference = data["conference"]

                # Recupera gli argomenti della conferenza
                topics = conference_topics.get(conference, "Nessun argomento disponibile")

                # Tokenizza il testo combinando titolo, abstract, conferenza e argomenti
                text = f"{title} [SEP] {abstract} [SEP] {conference} [SEP] {topics}"
                encoded = tokenizer(
                    text,
                    max_length=4096,  # Mistral supporta sequenze più lunghe
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                paper_ids.append(paper_id)
                input_ids.append(encoded["input_ids"].squeeze(0))
                attention_masks.append(encoded["attention_mask"].squeeze(0))

                # Converti la label in 0 (Non Rilevante) e 1 (Rilevante)
                relevance_label = 1 if data["relevance_label"] == "Rilevante" else 0
                labels.append(torch.tensor(relevance_label, dtype=torch.long))
            except Exception as e:
                print(f"Errore nella tokenizzazione di una riga: {e}")

    # Salva il dataset tokenizzato
    dataset = {
        "paper_ids": paper_ids,
        "input_ids": torch.stack(input_ids),
        "attention_masks": torch.stack(attention_masks),
        "labels": torch.tensor(labels)
    }

    torch.save(dataset, output_file)
    print(f"Dataset tokenizzato salvato in: {output_file}")

# Esegue la tokenizzazione
print("Tokenizzazione Training Set con argomenti conferenza...")
tokenize_dataset(train_file, train_output)

print("Tokenizzazione Test Set con argomenti conferenza...")
tokenize_dataset(test_file, test_output)

print("Tokenizzazione completata!")
