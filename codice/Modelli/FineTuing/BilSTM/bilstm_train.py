
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Configurazione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training su: {device}")


train_path = "/content/drive/MyDrive/ModelloTesi/train_tokenized_bilstm.pt"
test_path  = "/content/drive/MyDrive/ModelloTesi/test_tokenized_bilstm.pt"
model_path = "/content/drive/MyDrive/ModelloTesi/models/bilstm_model.pt"
metrics_path = "/content/drive/MyDrive/ModelloTesi/models/bilstm_metrics.json"

# Caricamento dati
train_data = torch.load(train_path)
test_data  = torch.load(test_path)

train_dataset = TensorDataset(train_data["input_ids"], train_data["labels"])
test_dataset  = TensorDataset(test_data["input_ids"], test_data["labels"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32)

# Calcolo class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data["labels"].numpy()),
    y=train_data["labels"].numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class Weights: {class_weights}")

# Modello BiLSTM
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden)
        return self.fc(out)

vocab_size = int(train_data["input_ids"].max().item()) + 1
model = BiLSTMClassifier(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

# Valutazione
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = [b.to(device) for b in batch]
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

metrics = {
    "accuracy": accuracy_score(all_labels, all_preds),
    "precision": precision_score(all_labels, all_preds, zero_division=0),
    "recall": recall_score(all_labels, all_preds, zero_division=0),
    "f1": f1_score(all_labels, all_preds, zero_division=0)
}

print("=== 📊 METRICHE ===")
print(json.dumps(metrics, indent=4))

# 📌 Salvataggi
torch.save(model.state_dict(), model_path)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Modello salvato in {model_path}")
print(f"Metriche salvate in {metrics_path}")
