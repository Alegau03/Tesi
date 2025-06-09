import os
import json
import torch
import numpy as np
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
from peft import get_peft_model, LoraConfig, TaskType

# Imposta cache locale e configurazioni CUDA
os.environ["HF_HOME"] = "/home/ubuntu/mnt-accessible/gautieri/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/ubuntu/mnt-accessible/gautieri/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 1. Configurazione
MODEL_NAME = "/home/ubuntu/mnt-accessible/gautieri/hf_cache/mistral_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training su: {device} | {torch.cuda.device_count()} GPU disponibili")

# 2. Caricamento dei dati tokenizzati
def load_tokenized_data(path):
    return torch.load(path)

train_data = load_tokenized_data("FileTokenizzati/tokenizzatiMistral/train_tokenized_mistral.pt")
test_data  = load_tokenized_data("FileTokenizzati/tokenizzatiMistral/test_tokenized_mistral.pt")

# 3. Calcolo dei pesi per le classi
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data["labels"].numpy()),
    y=train_data["labels"].numpy()
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
print(f"Class Weights: {class_weights}")

# 4. Conversione in HuggingFace Dataset
def convert_to_hf_dataset(data):
    return Dataset.from_dict({
        "input_ids": data["input_ids"].tolist(),
        "attention_mask": data["attention_masks"].tolist(),
        "labels": data["labels"].tolist()
    })

train_dataset = convert_to_hf_dataset(train_data)
test_dataset  = convert_to_hf_dataset(test_data)

# 5. Tokenizer e padding
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 6. Caricamento del modello
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16
).to(device)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

#model.gradient_checkpointing_enable()

# 7. Integrazione LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)
print("\n LoRA integrato: fine-tuning parameter-efficient abilitato.")

# 8. Metriche
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }

# 9. Trainer personalizzato
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 10. TrainingArguments
training_args = TrainingArguments(
    output_dir="./ModelliFinali/mistral_7b",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="adamw_torch_fused",
    seed=42,
    remove_unused_columns=False,
    report_to="none",
    max_grad_norm=1.0
)

# 11. Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 12. Training
print("\n Inizio fine-tuning di Mistral 7B su GPU L40S con LoRA e loss pesata...")
trainer.train()

# 13. Valutazione finale
print("\n Valutazione finale...")
metrics = trainer.evaluate()

# 14. Salvataggio
final_model_path = "./ModelliFinali/mistral_7b_final"
metrics_path = "./ModelliFinali/mistral_7b_metrics.json"
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
with open(metrics_path, "w") as f:
    json.dump({"test_metrics": metrics}, f, indent=4)

print(f"\n Modello salvato in {final_model_path}")
print(f" Metriche salvate in {metrics_path}")
