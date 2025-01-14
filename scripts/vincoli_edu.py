import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx


def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Funzione per analizzare gli EDUs
def get_all_edus_lengths(data):
    text_lengths = []
    for dialogue in data:
        for turn in dialogue.get('edus'):
            edu_text = turn.get('text')
            text_lengths.append(len(edu_text))
    return text_lengths

# Funzione che restituisce la lista di tutte le EDUs contenute nel dataset
def get_edus_list(data):
    unique_edus = set()
    for dialogue in data:
        for turn in dialogue.get('edus', []):
            edu_text = "[CLS] " + turn.get('text', "")
            unique_edus.add(edu_text)
    return list(unique_edus)

# Funzione che restituisce la lista delle EDU del primo dialogo del dataset
def get_edus_list_first(data):
    unique_edus = []
    for dialogue in data:
        for turn in dialogue.get('edus', []):
            edu_text = "[CLS] " + turn.get('text', "")
            unique_edus.append(edu_text)
        return list(unique_edus)

# Funzione che restituisce la lista di dialoghi contenuti nel dataset con il token [CLS] indicante l'inizio
# della stringa e i token [SEP] che contraddistinguono le singole EDU di ogni dialogo. Tali token sono
# utili per il modello BertTokenizer che effettua la tokenizzazione del testo su cui calcolare l'embedding
def get_edus_dialogue_list(data):
    edus_dialogue_list = []
    for element in data:
        dialogue = "[CLS] "
        for turn in element.get('edus'):
            dialogue += turn.get('text') + " [SEP] "

        parts = dialogue.rsplit(" [SEP] ", 1)

        edus_dialogue_list.append(parts[0])

    return edus_dialogue_list


# Funzione che restituisce una lista di coppie (t, c) in cui t rappresenta la singola EDU
# e c è il numero di volte che la EDU t appare nel dataset
def get_edus_counter_list(data):
    edu_counter = Counter()
    for dialogue in data:
        for turn in dialogue.get('edus', []):
            edu_text = turn.get('text', "")
            edu_counter[edu_text] += 1

    edu_list = sorted(edu_counter.items(), key=lambda item: item[1], reverse=True)

    return edu_list

def plot_top_edus(edu_list):
    top_20_edus = edu_list[:20]
    edus = [item[0] for item in top_20_edus]  # EDU text
    counts = [item[1] for item in top_20_edus]  # Frequenza

    plt.figure(figsize=(10, 6))
    plt.bar(edus, counts, color='skyblue')
    plt.xticks(rotation=90, fontsize=10)
    plt.xlabel("EDU", fontsize=12)
    plt.ylabel("Frequenza", fontsize=12)
    plt.title("EDUs più frequenti", fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    file_path = "dataset/STAC/train_subindex.json"
    data = load_data(file_path)

    edus_lengths = get_all_edus_lengths(data)
    edus_counter_list = get_edus_counter_list(data)

    print(f"Number of EDUs: {len(edus_lengths)}")
    print(f"Average EDU length: {sum(edus_lengths) / len(edus_lengths):.2f}")
    print(f"Number of unique EDUs: {len(edus_counter_list)}")
    # print("Sample of EDUs with counts:")
    # print(edus_counter_list[:20])

    plot_top_edus(edus_counter_list)

if __name__ == "__main__":
    main()
