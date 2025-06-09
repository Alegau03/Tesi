import os
import json
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  1. Configurazione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-large"
print(f" Training su: {device} | {torch.cuda.device_count()} GPU disponibili")

#  2. Percorsi
TRAIN_PATH = "FileTokenizzati/tokenizzatiDeBERTa/train_tokenized_deberta.pt"
TEST_PATH = "FileTokenizzati/tokenizzatiDeBERTa/test_tokenized_deberta.pt"
OUTPUT_DIR = "./ModelliFinali/deberta_v3_large"

#  3. Caricamento dataset tokenizzati
def load_tokenized_data(path):
    return torch.load(path)

train_data = load_tokenized_data(TRAIN_PATH)
test_data = load_tokenized_data(TEST_PATH)

#  4. Calcolo pesi delle classi per bilanciamento
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data["labels"].numpy()),
    y=train_data["labels"].numpy()
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
print(f" Class Weights: {class_weights}")

#  5. Conversione in Dataset Hugging Face
def to_hf_dataset(data):
    return Dataset.from_dict({
        "input_ids": data["input_ids"].tolist(),
        "attention_mask": data["attention_masks"].tolist(),
        "labels": data["labels"].tolist()
    })

train_dataset = to_hf_dataset(train_data)
test_dataset = to_hf_dataset(test_data)

#  6. Caricamento del modello
model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16  # Ottimizzato per GPU L40S
).to(device)

# Abilita il gradient checkpointing per ridurre l'uso della memoria
model.gradient_checkpointing_enable()



#  7. Definizione delle metriche di valutazione
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }

#  8. Custom Trainer per utilizzare la loss pesata
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

#  9. TrainingArguments ottimizzati per GPU L40S (46 GB VRAM)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Batch effettivo: 16 * 2 = 32
    num_train_epochs=10,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs_deberta",
    logging_steps=100,
    save_total_limit=2,
    bf16=True,              # Abilita BF16, ottimale per L40S
    fp16=False,
    optim="adamw_torch_fused",  # Ottimizzatore fused per migliori performance
    seed=42,
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=1.0        # Clipping dei gradienti per stabilità
)

#  10. Preparazione del tokenizer e data_collator
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer)

#  11. Inizializzazione del Trainer (usando il CustomTrainer per la loss pesata)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

#  12. Avvio del fine-tuning
print(" Inizio fine-tuning di DeBERTa V3 Large su GPU L40S...")
trainer.train()

# 13. Valutazione finale
print(" Valutazione finale...")
metrics = trainer.evaluate()

#14. Salvataggio finale del modello e delle metriche
model_final_path = os.path.join(OUTPUT_DIR, "final_model")
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
os.makedirs(model_final_path, exist_ok=True)
model.save_pretrained(model_final_path)
tokenizer.save_pretrained(model_final_path)

with open(metrics_path, "w") as f:
    json.dump({"test_metrics": metrics}, f, indent=4)

print(f"Modello salvato in {model_final_path}")
print(f"Metriche salvate in {metrics_path}")
print("Fine-tuning DeBERTa completato!")
