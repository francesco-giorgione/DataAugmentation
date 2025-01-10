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
            edu_text = turn.get('text', "")
            unique_edus.add(edu_text)
    return list(unique_edus)

# Funzione che restituisce una lista di coppie (t, c) in cui t rappresenta la singola EDU
# e c è il numero di volte che la EDU t appare nel dataset
def get_edus_counter_list(data):
    edu_counter = Counter()
    for dialogue in data:
        for turn in dialogue.get('edus', []):
            edu_text = turn.get('text', "")
            edu_counter[edu_text] += 1
    return sorted(edu_counter.items(), key=lambda item: item[1], reverse=True)


def main():
    file_path = "dataset/STAC/train_subindex.json"
    data = load_data(file_path)

    edus_lengths = get_all_edus_lengths(data)
    edus_counter_list = get_edus_counter_list(data)

    print(f"Number of EDUs: {len(edus_lengths)}")
    print(f"Average EDU length: {sum(edus_lengths) / len(edus_lengths):.2f}")
    print(f"Number of unique EDUs: {len(edus_counter_list)}")
    print("Sample of EDUs with counts:")
    print(edus_counter_list[:785])

if __name__ == "__main__":
    main()
