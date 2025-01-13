import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx
import json
from collections import defaultdict

import numpy as np

def frequenze_relazioni(filename, dict_relazioni):
    with open(filename, 'r') as file:
        data = json.load(file)

    for session in data:
        relazioni = session.get("relations", [])
        tipi_relazioni = [diz['type'] for diz in relazioni if 'type' in diz]

        for id in range(len(tipi_relazioni)):
            if id+1 < len(tipi_relazioni):
                dict_relazioni[(tipi_relazioni[id], tipi_relazioni[id+1])] += 1

    return dict_relazioni


def create_stochastic_matrix(dict_relazioni, name_dataset):
    # Creo una lista di relazioni presenti nel dizionario
    states = list(set([k[0] for k in dict_relazioni.keys()] + [k[1] for k in dict_relazioni.keys()]))

    # Inizzializzo la matrice
    matrix = pd.DataFrame(0, index=states, columns=states)
    for (state1, state2), count in dict_relazioni.items():
        matrix.loc[state1, state2] = count


    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        matrix,
        annot=True,  # Mostra le frequenze come annotazioni
        fmt="d",  # Formattazione come interi
        cmap="viridis",  # Mappa di colori
        xticklabels=matrix.columns,  # Etichette asse x
        yticklabels=matrix.index,  # Etichette asse y
        cbar_kws={"label": "Frequencies"},  # Etichetta per la barra colori
        ax=ax
    )

    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title("Frequency Matrix", pad=20)

    # Salvataggio dell'immagine
    plt.tight_layout()
    plt.savefig(f"graphic/{name_dataset}_freq_matrix.png")

    # Calcolo della matrice delle probabilità condizionate
    row_sums = matrix.sum(axis=1).replace(0, np.nan)  # Sostituisce eventuali somme zero con NaN per evitare divisioni per zero
    probability_matrix = matrix.div(row_sums, axis=0)  # Divisione riga per riga

    # Creazione della heatmap per la matrice delle probabilità condizionate
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        probability_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=matrix.columns,
        yticklabels=matrix.index,
        cbar_kws={"label": "P(x|y)"},
        ax=ax
    )

    ax.set_title("Conditional Probability Matrix P(x|y)", pad=20)
    ax.set_xlabel("x (Target State)")
    ax.set_ylabel("y (Source State)")

    plt.tight_layout()
    plt.savefig(f"graphic/{name_dataset}_stochastic_matrix.png")



if __name__ == "__main__":
    path_file1 = ['dataset/STAC/train_subindex.json']
    path_file2 = ['dataset/MOLWENI/train.json']
    path_file3 = ['dataset/MINECRAFT/TRAIN_307_bert.json']
    dict_relazioni1 = defaultdict(int)
    dict_relazioni2 = defaultdict(int)
    dict_relazioni3 = defaultdict(int)

    
    # STAC
    for file in path_file1:
        dict_relazioni1 = frequenze_relazioni(file, dict_relazioni1)
    
    create_stochastic_matrix(dict_relazioni1, 'STAC')

    # MOLWENI
    for file in path_file2:
        dict_relazioni2 = frequenze_relazioni(file, dict_relazioni2)
    
    create_stochastic_matrix(dict_relazioni2, 'MOLWENI')

    # MINECRAFT
    for file in path_file3:
        dict_relazioni3 = frequenze_relazioni(file, dict_relazioni3)
    
    create_stochastic_matrix(dict_relazioni3, 'MINECRAFT')


