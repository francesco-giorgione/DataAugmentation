import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.constraints.graph_builder import *


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
    file_path = None

    if dataset_name == "STAC_training":
        file_path = "../../dataset/STAC/train_subindex.json"
    elif dataset_name == "MOLWENI_training":
        file_path = "../../dataset/MOLWENI/train.json"
    elif dataset_name == "MINECRAFT_training":
        file_path = "../../dataset/MINECRAFT/TRAIN_307_bert.json"

    elif dataset_name == "STAC_testing":
        file_path = "../../dataset/STAC/test_subindex.json"
    elif dataset_name == "MOLWENI_testing":
        file_path = "../../dataset/MOLWENI/test.json"
    elif dataset_name == "MINECRAFT_testing":
        file_path = "../../dataset/MINECRAFT/TEST_133.json"
    
    elif dataset_name == "STAC_val":
        file_path = "../../dataset/STAC/dev.json"
    elif dataset_name == "MOLWENI_val":
        file_path = "../../dataset/MOLWENI/dev.json"
    elif dataset_name == "MINECRAFT_val":
        file_path = "../../dataset/MINECRAFT/VAL_all.json"

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

        subdialogue_list = get_subdialogue_list(data)
        print(subdialogue_list[:5])

if __name__ == "__main__":
    main()
