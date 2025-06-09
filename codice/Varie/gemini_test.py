
import google.generativeai as genai
import json
import os
import re
import time  # Importa il modulo time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm # Per mostrare una barra di avanzamento




API_KEY = "AIzaSyBpZLdqZMHENzwtJEqTqDKK0P4gSkUAJt4" 
if not API_KEY:
    print("Errore: Devi impostare la tua chiave API di Gemini.")
    exit()
    

genai.configure(api_key=API_KEY)


MODEL_NAME = 'gemini-1.5-flash-latest' 
PAPER_FILE = 'test_dataset.jsonl'
CONFERENCE_FILE = 'ArgomentiConferenze.jsonl'

# --- Caricamento Dati ---

def load_jsonl(filepath):
    """Carica un file JSONL in una lista di dizionari."""
    data = []
    try:
        
        df = pd.read_json(filepath, lines=True)
        # Converte il DataFrame in una lista di dizionari
        data = df.to_dict('records')
    except FileNotFoundError:
        print(f"Errore: File non trovato: {filepath}")
        return None
    except ValueError as e:
        print(f"Errore durante la lettura del JSONL con pandas: {e}")
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as json_e:
                        print(f"Errore nel decodificare la riga {i+1}: {line.strip()} - {json_e}")
                        continue # Salta la riga malformattata
        except Exception as fallback_e:
             print(f"Errore anche nel tentativo di fallback: {fallback_e}")
             return None # Fallito caricamento
    return data

print(f"Caricamento paper da {PAPER_FILE}...")
papers_data = load_jsonl(PAPER_FILE)
print(f"Caricamento argomenti conferenze da {CONFERENCE_FILE}...")
conference_topics_list = load_jsonl(CONFERENCE_FILE)

if papers_data is None or conference_topics_list is None:
    print("Errore nel caricamento dei file. Interruzione dello script.")
    exit()

print(f"Caricati {len(papers_data)} paper.")
print(f"Caricati {len(conference_topics_list)} record di conferenze.")

# Preprocessa gli argomenti delle conferenze per una ricerca rapida
# Crea una chiave unica (nome_conferenza_normalizzato, anno_stringa)
conference_topics_map = {}
for conf in conference_topics_list:
    conf_name = conf.get('Conferenza', '').strip()
    conf_year = str(conf.get('Anno', '')).strip()
    if conf_name and conf_year:
        # Normalizza nome conferenza (maiuscolo, rimuove spazi extra)
        normalized_conf_name = ' '.join(conf_name.split()).upper()
        conference_topics_map[(normalized_conf_name, conf_year)] = conf.get('Argomenti', [])
    else:
         print(f"Attenzione: Record conferenza saltato per mancanza di nome o anno: {conf}")

print(f"Mappa argomenti conferenze creata con {len(conference_topics_map)} voci.")

# --- Preparazione Modello Gemini ---

# Configurazione per la generazione di testo (per la classificazione)
generation_config = genai.types.GenerationConfig(
    max_output_tokens=20 # Aumentato leggermente per sicurezza
)

# Configurazione di sicurezza 
safety_settings = {
     "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
     "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
     "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
     "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
}


model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- Funzione per creare il Prompt ---

def create_prompt(paper_title, paper_abstract, conference_name, conference_year, topics):
    """Crea il prompt per Gemini."""
    topic_list_str = ", ".join(topics) if topics else "Nessun argomento specificato"
    prompt = f"""
    **Contesto:** Sei un assistente AI che valuta la pertinenza di un paper scientifico rispetto agli argomenti specifici di una conferenza.

    **Paper da Valutare:**
    * **Titolo:** "{paper_title}"
    * **Abstract:** {paper_abstract}

    **Conferenza di Riferimento:**
    * **Nome:** {conference_name}
    * **Anno:** {conference_year}
    * **Argomenti Principali della Conferenza ({conference_name} {conference_year}):** {topic_list_str}

    **Compito:**
    Analizza attentamente il **titolo** e l'**abstract** del paper. Determina se il contenuto principale del paper è **strettamente correlato** e **pertinente** agli argomenti principali elencati per la conferenza {conference_name} {conference_year}.

    **Istruzioni per la Risposta:**
    Rispondi **SOLO** con una delle due parole seguenti:
    * `Rilevante` (se il paper è pertinente agli argomenti della conferenza)
    * `Non rilevante` (se il paper NON è pertinente agli argomenti della conferenza)

    Non aggiungere spiegazioni o altre parole.
    """
    return prompt

# --- Classificazione ed Valutazione ---

true_labels = []
predicted_labels = []
failed_papers = [] # Tiene traccia dei paper che non è stato possibile classificare

print(f"\nInizio classificazione con il modello {MODEL_NAME}...")
# Aggiornato per riflettere il limite di 15 RPM
print(f"Verrà applicata una pausa di ~4.1 secondi tra ogni chiamata API per rispettare il limite di 15 RPM.")
print(f"Tempo stimato per il completamento: circa {1270 * 4.1 / 60:.1f} minuti.")

# Usiamo tqdm per una barra di progresso
for paper in tqdm(papers_data, desc="Classificazione Paper"):
    paper_id = paper.get('paper_id', 'ID_Sconosciuto')
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    conference_full_name = paper.get('conference', '')
    true_label = paper.get('relevance_label', '').strip().capitalize()

    if not title or not abstract or not conference_full_name or not true_label:
        print(f"\nAttenzione: Paper {paper_id} saltato per dati mancanti.")
        failed_papers.append({'id': paper_id, 'reason': 'Dati mancanti nel JSONL'})
        continue

    if true_label not in ["Rilevante", "Non rilevante"]:
         print(f"\nAttenzione: Etichetta di rilevanza non valida per paper {paper_id}: '{true_label}'. Saltato.")
         failed_papers.append({'id': paper_id, 'reason': f"Etichetta di rilevanza non valida: '{true_label}'"})
         continue

    match = re.match(r'^(.*?)\s+(\d{4})$', conference_full_name.strip())
    if not match:
         print(f"\nAttenzione: Formato conferenza non riconosciuto per paper {paper_id}: '{conference_full_name}'. Saltato.")
         failed_papers.append({'id': paper_id, 'reason': f"Formato conferenza non riconosciuto: '{conference_full_name}'"})
         continue

    conf_name_extracted = match.group(1).strip()
    conf_year_extracted = match.group(2).strip()
    normalized_conf_name_extracted = ' '.join(conf_name_extracted.split()).upper()

    conference_key = (normalized_conf_name_extracted, conf_year_extracted)
    topics = conference_topics_map.get(conference_key)

    if topics is None:
        print(f"\nAttenzione: Argomenti non trovati per {normalized_conf_name_extracted} {conf_year_extracted} (Paper: {paper_id}). Classifico senza contesto argomenti.")
        topics = []

    prompt = create_prompt(title, abstract, conf_name_extracted, conf_year_extracted, topics)

    try:
        response = model.generate_content(prompt)
        prediction_text = response.text.strip().lower()

        if "non rilevante" in prediction_text:
            predicted_label = "Non rilevante"
        elif "rilevante" in prediction_text:
             predicted_label = "Rilevante"
        else:
            print(f"\nAttenzione: Risposta inattesa dal modello per paper {paper_id}: '{response.text.strip()}'. Tento di interpretare...")
            if "rilevante" in prediction_text:
                 predicted_label = "Rilevante"
                 print(f"Interpretato come 'Rilevante'")
            elif "non" in prediction_text:
                 predicted_label = "Non rilevante"
                 print(f"Interpretato come 'Non rilevante'")
            else:
                 predicted_label = "Non rilevante"
                 print(f"Interpretazione fallita. Classificato come 'Non rilevante' (default).")
                 failed_papers.append({'id': paper_id, 'reason': f"Risposta modello non interpretabile: '{response.text.strip()}'"})

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

      
        time.sleep(4.1) #Aggiungiamo del tempo di attesa per rispettare il limite di 15 RPM

    except Exception as e:
        error_message = str(e)
        print(f"\nErrore durante la chiamata API per paper {paper_id}: {error_message}")
        failed_papers.append({'id': paper_id, 'reason': f"Errore API: {error_message}"})

        if "429" in error_message or "quota" in error_message.lower() or "rate limit" in error_message.lower():
            print("Rate limit raggiunto. Attendo 60 secondi prima di riprovare...")
            time.sleep(60)
        elif "safety settings" in error_message.lower():
             print("Contenuto bloccato dalle safety settings. Il paper non verrà classificato.")
        else:
             print("Attendo 5 secondi prima di continuare...")
             time.sleep(5)

# --- Calcolo Metriche ---

print("\nClassificazione completata.")

if failed_papers:
    print(f"\nAttenzione: Non è stato possibile classificare o interpretare correttamente {len(failed_papers)} paper.")

if not true_labels or not predicted_labels:
    print("\nNessun paper classificato con successo o etichette non consistenti. Impossibile calcolare le metriche.")
else:
    print(f"\nCalcolo metriche su {len(true_labels)} paper classificati con successo...")

    possible_labels = ["Rilevante", "Non rilevante"]
    valid_indices = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels)) if t in possible_labels and p in possible_labels]

    if len(valid_indices) != len(true_labels):
        print(f"Attenzione: {len(true_labels) - len(valid_indices)} coppie di etichette non valide sono state escluse dalla valutazione.")
        true_labels_eval = [true_labels[i] for i in valid_indices]
        predicted_labels_eval = [predicted_labels[i] for i in valid_indices]
    else:
        true_labels_eval = true_labels
        predicted_labels_eval = predicted_labels

    if not true_labels_eval:
         print("Nessuna etichetta valida rimasta dopo il filtraggio. Impossibile calcolare le metriche.")
    else:
        positive_label = "Rilevante"
        accuracy = accuracy_score(true_labels_eval, predicted_labels_eval)
        precision = precision_score(true_labels_eval, predicted_labels_eval, pos_label=positive_label, zero_division=0, labels=possible_labels)
        recall = recall_score(true_labels_eval, predicted_labels_eval, pos_label=positive_label, zero_division=0, labels=possible_labels)
        f1 = f1_score(true_labels_eval, predicted_labels_eval, pos_label=positive_label, zero_division=0, labels=possible_labels)

        print("\n--- Risultati Valutazione ---")
        print(f"Numero di Paper Valutati Effettivamente: {len(true_labels_eval)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (per classe '{positive_label}'): {precision:.4f}")
        print(f"Recall (per classe '{positive_label}'): {recall:.4f}")
        print(f"F1-Score (per classe '{positive_label}'): {f1:.4f}")
        print(f"Eval Loss: Non calcolabile direttamente con questo metodo.")

        try:
            from sklearn.metrics import classification_report
            print("\nReport di Classificazione Dettagliato:")
            print(classification_report(true_labels_eval, predicted_labels_eval, labels=possible_labels, zero_division=0))
        except ImportError:
            print("\n(Installa scikit-learn con 'pip install scikit-learn' per vedere il report di classificazione dettagliato)")

print("\nScript completato.")