# Installazione delle dipendenze necessarie
!pip install transformers accelerate sentencepiece bitsandbytes > /dev/null

import os
import json
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# Percorsi dei dataset
DATASET_FILES = ["cleaned_reviews.jsonl", "cleaned_peersum.jsonl"]
OUTPUT_FILE = "llama_classification_results.jsonl"

# Modello da utilizzare (più leggero per GPU T4)
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


# 1. Disattiva i warning per evitare spam in output
warnings.filterwarnings("ignore")

# 2. Caricamento del Modello con Offload ottimizzato per GPU T4
def load_llama_instruct():
    print(f"Caricamento del modello: {MODEL_NAME} ...")

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Usa quantizzazione a 4-bit per risparmiare memoria
            bnb_4bit_compute_dtype=torch.float16,  # Precisione 16-bit
            llm_int8_enable_fp32_cpu_offload=True  # Sposta moduli extra sulla CPU se necessario
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",  # Distribuzione automatica tra GPU e CPU
            offload_folder="offload_dir"  # Offload su disco per moduli fuori VRAM
        )

        print("Modello caricato correttamente! Inizio elaborazione...")
        textgen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # 🔹 Evita warning su padding token
        )
        return textgen_pipeline, tokenizer

    except Exception as e:
        print(f" Errore nel caricamento di {MODEL_NAME}: {e}")
        return None, None

# 3. Creazione del Prompt per LLaMA
def build_prompt(review_text, accepted, tokenizer):
    if not isinstance(review_text, str) or review_text.strip() == "":
        review_text = "No review provided."

    system_msg = (
        "You are an AI assistant that analyzes academic peer reviews to determine whether a paper is relevant to the conference.\n"
        "Rules to follow:\n"
        "- If the paper was accepted, classify it as 'Rilevante'.\n"
        "- If the paper was rejected, analyze the review:\n"
        "  - If the review states that the paper is not relevant or suggests a different track/workshop, classify it as 'Non rilevante'.\n"
        "  - If the paper was rejected for methodological flaws but remains within the scope of the conference, classify it as 'Rilevante'.\n"
        "  - If the review does not provide enough information to decide, classify it as 'Non determinato'.\n"
        "\n"
        "IMPORTANT: Respond with ONLY ONE of these exact words:\n"
        "- Rilevante\n"
        "- Non rilevante\n"
        "- Non determinato\n"
    )

    review_tokens = tokenizer.encode(review_text, truncation=True, max_length=512)
    review_text = tokenizer.decode(review_tokens)

    user_msg = f"Review:\n{review_text}\n\nIs this paper relevant to the conference?"
    prompt = f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n<<USER>>\n{user_msg}\n<</USER>>"

    return prompt

# 4. Analisi e Classificazione della Rilevanza
def classify_relevance(review_text, accepted, textgen_pipeline, tokenizer):
    #Tutti i paper accettati sono automaticamente "Rilevante"
    if accepted == 1:
        return "Rilevante"

    # Paper rifiutati vengono analizzati da LLaMA
    prompt = build_prompt(review_text, accepted, tokenizer)
    output = textgen_pipeline(prompt, max_new_tokens=50)

    result_text = output[0]["generated_text"].strip().lower()

    if "non rilevante" in result_text:
        return "Non rilevante"
    elif "rilevante" in result_text:
        return "Rilevante"
    elif "non determinato" in result_text:
        return "Non determinato"
    else:
        return "Non determinato"

# 5. Processa il Dataset con Stampa Ogni 100 Paper
def process_jsonl(dataset_files, output_file, textgen_pipeline, tokenizer):
    results = []
    count_rilevante = 0
    count_non_rilevante = 0
    count_non_determinato = 0

    for dataset_file in dataset_files:
        if not os.path.exists(dataset_file):
            print(f" ERRORE: File {dataset_file} non trovato. Skippato.")
            continue

        print(f" Elaborazione di {dataset_file}")
        with open(dataset_file, "r", encoding="utf-8") as f_in:
            lines = [json.loads(line.strip()) for line in f_in.readlines()]

        for idx, data in enumerate(lines):
            paper_id = data.get("paper_id", "unknown")
            review_text = data.get("review_text", "")
            accepted = data.get("accepted", None)  # 1 = Accettato, 0 = Rifiutato, None = Non Determinato

            label = classify_relevance(review_text, accepted, textgen_pipeline, tokenizer)
            results.append({
                "paper_id": paper_id,
                "relevance_label": label
            })

            # Stampa ogni 100 paper elaborati
            if idx % 100 == 0:
                print(f" {idx}/{len(lines)} paper valutati... Ultimo paper processato: {paper_id}")

    # Salva i risultati
    with open(output_file, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== RISULTATI FINALI ===")
    print(f" Rilevanti: {count_rilevante}")
    print(f" Non Rilevanti: {count_non_rilevante}")
    print(f" Non Determinati: {count_non_determinato}")
    print(f" Risultati salvati in {output_file}")

# 6. Esegui il programma
def main():
    textgen_pipeline, tokenizer = load_llama_instruct()
    if textgen_pipeline and tokenizer:
        process_jsonl(DATASET_FILES, OUTPUT_FILE, textgen_pipeline, tokenizer)
    else:
        print(" Errore: Modello non caricato correttamente.")

if __name__ == "__main__":
    main()
