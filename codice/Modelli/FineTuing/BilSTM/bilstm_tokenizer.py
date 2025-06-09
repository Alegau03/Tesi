import json
import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence


train_file = "/content/drive/MyDrive/ModelloTesi/train_dataset.jsonl"
test_file = "/content/drive/MyDrive/ModelloTesi/test_dataset.jsonl"
conference_topics_file = "/content/drive/MyDrive/ModelloTesi/ArgomentiConferenze.jsonl"
train_output = "/content/drive/MyDrive/ModelloTesi/train_tokenized_bilstm.pt"
test_output = "/content/drive/MyDrive/ModelloTesi/test_tokenized_bilstm.pt"

# Tokenizer BERT (solo per word embedding)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Carica argomenti conferenza
conference_topics = {}
with open(conference_topics_file, "r") as f:
    for line in f:
        data = json.loads(line.strip())
        key = f"{data['Conferenza']} {data['Anno']}"
        conference_topics[key] = ", ".join(data["Argomenti"])

# Funzione per tokenizzare
def tokenize_dataset(json_file, output_file):
    input_ids, labels = [], []
    with open(json_file, "r") as f:
        for line in f:
            data = json.loads(line)
            text = f"{data['title']} [SEP] {data['abstract']} [SEP] {data['conference']} [SEP] {conference_topics.get(data['conference'], '')}"
            tokens = tokenizer.encode(text, truncation=True, max_length=512, add_special_tokens=True)
            input_ids.append(torch.tensor(tokens, dtype=torch.long))
            labels.append(1 if data["relevance_label"] == "Rilevante" else 0)
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.tensor(labels)
    torch.save({"input_ids": input_ids, "labels": labels}, output_file)
    print(f"Salvato: {output_file}")


tokenize_dataset(train_file, train_output)
tokenize_dataset(test_file, test_output)
