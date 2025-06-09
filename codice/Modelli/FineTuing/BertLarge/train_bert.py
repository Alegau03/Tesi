import os
import json
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configurazione GPU L40S
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-large-uncased"
print(f"Training su: {device} | VRAM disponibile: ~46 GB (GPU NVIDIA L40S)")

# 2. Caricamento file tokenizzati
def load_tokenized_data(path):
    return torch.load(path)

train_data = load_tokenized_data("../../FileTokenizzati/tokenizzatiBert/train_tokenized.pt")
test_data = load_tokenized_data("../../FileTokenizzati/tokenizzatiBert/test_tokenized.pt")

# 3. Calcolo pesi delle classi per bilanciamento
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data["labels"].numpy()),
    y=train_data["labels"].numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class Weights: {class_weights}")

# 4. Conversione in Dataset HuggingFace
def convert_to_dataset(data):
    return Dataset.from_dict({
        "input_ids": data["input_ids"].tolist(),
        "attention_mask": data["attention_masks"].tolist(),
        "labels": data["labels"].tolist()
    })

train_dataset = convert_to_dataset(train_data)
test_dataset = convert_to_dataset(test_data)

# 5. Caricamento del modello BERT Large
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)


# model.gradient_checkpointing_enable()  # VRAM abbondante quindi commentato

# 📌 6. Definizione delle metriche di valutazione
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary")
    }

# 7. Custom Trainer per utilizzare i class_weights nella loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 8. TrainingArguments ottimizzati per L40S (46 GB VRAM)
training_args = TrainingArguments(
    output_dir="./models/bert_large_l40s",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,    # Batch ampio, grazie alla VRAM abbondante
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,       # Effettivo batch size: 16 * 4 = 64
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,                         # Usa bfloat16 su GPU L40S
    fp16=False,
    optim="adamw_torch_fused",         # Ottimizzatore fused
    seed=42,
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=1.0                   # Clipping dei gradienti per maggiore stabilità
)

#9. Preparazione del tokenizer e del data_collator
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer)

# 10. Inizializzazione del Trainer personalizzato
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 11. Fine-tuning
print("Inizio fine-tuning BERT Large su GPU L40S...")
trainer.train()

# 12. Valutazione finale
print("Valutazione finale...")
metrics = trainer.evaluate()

# 13. Salvataggio del modello e delle metriche
final_model_path = "./models/bert_large_l40s_final"
metrics_path = "./models/bert_large_l40s_metrics.json"
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

with open(metrics_path, "w") as f:
    json.dump({"test_metrics": metrics}, f, indent=4)

print(f"Modello salvato in {final_model_path}")
print(f"Metriche salvate in {metrics_path}")
