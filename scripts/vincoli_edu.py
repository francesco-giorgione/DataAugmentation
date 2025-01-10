import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx

# Funzione per caricare il dataset STAC
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Funzione per analizzare gli EDUs
def analyze_edus(data):
    edu_lengths = []
    text_lengths = []
    for dialogue in data:
        for turn in dialogue.get('edus'):
            edu_text = turn.get('text')
            text_lengths.append(len(edu_text))
            # for edu in edus:
            #     edu_lengths.append(len(edu.split()))
    return text_lengths

# Funzione per ottenere EDUs uniche
def get_unique_edus(data):
    unique_edus = set()
    for dialogue in data:
        for turn in dialogue.get('edus', []):
            edu_text = turn.get('text', "")
            unique_edus.add(edu_text)
    return list(unique_edus)

# Funzione per ottenere EDUs uniche con un contatore
def get_edus_with_counts(data):
    edu_counter = Counter()
    for dialogue in data:
        for turn in dialogue.get('edus', []):
            edu_text = turn.get('text', "")
            edu_counter[edu_text] += 1
    return sorted(edu_counter.items(), key=lambda item: item[1], reverse=True)

# Main function to perform analysis
def main():
    file_path = "dataset/STAC/train_subindex.json"
    data = load_data(file_path)

    edu_lengths = analyze_edus(data)
    edus_with_counts = get_edus_with_counts(data)

    print(f"Number of EDUs: {len(edu_lengths)}")
    print(f"Average EDU length: {sum(edu_lengths) / len(edu_lengths):.2f}")
    print(f"Number of unique EDUs: {len(edus_with_counts)}")
    print("Sample of EDUs with counts:")
    print(edus_with_counts[:785])

if __name__ == "__main__":
    main()
