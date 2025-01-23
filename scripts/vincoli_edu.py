import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from scripts.graph_builder import *


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
    unique_edus = []
    for dialogue in data:
        for turn in dialogue.get('edus', []):
            edu_text = turn.get('text', "")
            if edu_text not in unique_edus:  # Controlla se l'elemento è già presente
                unique_edus.append(edu_text)
    return unique_edus


def get_dialogue_edus_list(data):
    dialogue_edus_list = []
    for element in data:
        edus = []
        for turn in element.get('edus'):
            edus.append(turn.get('text'))

        dialogue_edus_list.append(edus)

    return dialogue_edus_list

# # Funzione che restituisce la lista delle EDU del primo dialogo del dataset
# def get_edus_list_first(data):
#     unique_edus = []
#     for dialogue in data:
#         for turn in dialogue.get('edus', []):
#             edu_text = turn.get('text', "")
#             unique_edus.append(edu_text)
#         return list(unique_edus)

# # Funzione che restituisce la lista di dialoghi contenuti nel dataset con il token [CLS] indicante l'inizio
# # della stringa e i token [SEP] che contraddistinguono le singole EDU di ogni dialogo. Tali token sono
# # utili per il modello BertTokenizer che effettua la tokenizzazione del testo su cui calcolare l'embedding
# def get_edus_dialogue_list(data):
#     edus_dialogue_list = []
#     for element in data:
#         dialogue = "[CLS] "
#         for turn in element.get('edus'):
#             dialogue += turn.get('text') + " [SEP] "

#         parts = dialogue.rsplit(" [SEP] ", 1)

#         edus_dialogue_list.append(parts[0])

#     return edus_dialogue_list


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

def plot_top_edus(edu_list, dataset_name):
    top_10_edus = edu_list[:10]
    edus = [item[0] for item in top_10_edus]  # EDU text
    counts = [item[1] for item in top_10_edus]  # Frequenza

    plt.figure(figsize=(12, 8))
    plt.bar(edus, counts, color='skyblue')
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel("EDU", fontsize=10)
    plt.ylabel("Frequenza", fontsize=10)
    plt.title(dataset_name + "- EDUs più frequenti", fontsize=14)
    plt.tight_layout()

    plt.savefig(f"edus_analysis/{dataset_name}_top_edus.png")
    #lt.show()

def get_subdialogue_list(data):

    subdialogue_list = []

    for i, dialogo in enumerate(data):
        print(f"Creando il grafo per il dialogo {i + 1}...")
        grafo = crea_grafo_da_json([dialogo])  # Passiamo il singolo dialogo come lista

            # Convertire gli indici delle componenti connesse in EDUs
        result = []
        for nodo in grafo.nodes:
            text = grafo.nodes[nodo]["text"]
            result.append(text)
        
        subdialogue_list.append(result)
    
    return subdialogue_list


def get_filepath(dataset_name):

    if dataset_name == "STAC_training":
        file_path = "dataset/STAC/train_subindex.json"
    elif dataset_name == "MOLWENI_training":
        file_path = "dataset/MOLWENI/train.json"
    elif dataset_name == "MINECRAFT_training":
        file_path = "dataset/MINECRAFT/TRAIN_307_bert.json"

    elif dataset_name == "STAC_testing":
        file_path = "dataset/STAC/test_subindex.json"
    elif dataset_name == "MOLWENI_testing":
        file_path = "dataset/MOLWENI/test.json"
    elif dataset_name == "MINECRAFT_testing":
        file_path = "dataset/MINECRAFT/TEST_133.json"
    
    elif dataset_name == "STAC_val":
        file_path = "dataset/STAC/dev.json"
    elif dataset_name == "MOLWENI_val":
        file_path = "dataset/MOLWENI/dev.json"
    elif dataset_name == "MINECRAFT_val":
        file_path = "dataset/MINECRAFT/VAL_all.json"

    return file_path


def main():
    
    dataset_name_list = ["STAC_val"]

    for dataset_name in dataset_name_list:
        file_path = get_filepath(dataset_name)
    
        data = load_data(file_path)

        # edus_lengths = get_all_edus_lengths(data)
        
        # edus_counter_list = get_edus_counter_list(data)

        
        # print(f"Numbero di EDUs del dataset " + dataset_name + f": {len(edus_lengths)}")
        # print(f"Lunghezza media delle EDU del dataset " + dataset_name + f": {sum(edus_lengths) / len(edus_lengths):.2f}")
        # print(f"Number of EDUs senza ripetizioni " + dataset_name + f": {len(edus_counter_list)}")
        
        # print("Top 10 EDU più frequenti e relativa frequenza:")
        # print(edus_counter_list[:10])


        # plot_top_edus(edus_counter_list, dataset_name)

        subdialogue_list = get_subdialogue_list(data)
        print(subdialogue_list[:5])

if __name__ == "__main__":
    main()
