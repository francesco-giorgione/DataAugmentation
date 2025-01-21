import json
import networkx as nx
import matplotlib.pyplot as plt
import os

# Funzione per creare il grafo dal JSON
def crea_grafo_da_json(dialogo):
    G = nx.DiGraph()  # Creiamo un grafo diretto
    id_nodo = 0

    # Aggiungiamo i nodi (ogni EDU è un nodo)
    for edu in dialogo["edus"]:
        #id_nodo = edu["speechturn"]
        G.add_node(id_nodo, text = edu["text"], speaker = edu["speaker"])
        id_nodo += 1

    # Aggiungiamo le relazioni (archi tra i nodi)
    for relazione in dialogo["relations"]:
        x = relazione["x"]
        y = relazione["y"]
        relazione_tipo = relazione["type"]
        G.add_edge(x, y, relationship=relazione_tipo)  # Aggiungiamo l'arco con la relazione

    return G

# Funzione per visualizzare il grafo in forma di DAG verticale
def visualizza_grafo_dag(G, id_dialogo):
    try:
        # Usa Graphviz per generare un layout coerente per il DAG
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Usa il programma `dot` per layout verticale
    except ImportError:
        # Fallback al layout manuale se Graphviz non è disponibile
        pos = nx.spring_layout(G)

    # Disegno del grafo
    plt.figure(figsize=(10, 8))  # Dimensione personalizzata
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8, arrows=True, arrowsize=15)

    # Etichette per gli archi (relazioni)
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False, label_pos=0.5)

    # Titolo del grafo
    plt.title(f"Grafo del Dialogo {id_dialogo} (DAG)", fontsize=14)

    # Mostra il grafo
    plt.tight_layout()
    plt.show()

# Funzione per esportare gli utterances in un file di testo
def esporta_utterances_singolo(dialogo, id_dialogo, cartella_output="output"):
    # Creiamo la cartella di output se non esiste
    os.makedirs(cartella_output, exist_ok=True)

    nome_file_output = os.path.join(cartella_output, f"dialogo_{id_dialogo}.txt")
    with open(nome_file_output, 'w') as f:
        id_nodo = 0
        f.write(f"Dialogo {id_dialogo}:\n")
        for edu in dialogo["edus"]:
            #id_nodo = edu["speechturn"]
            speaker = edu["speaker"]
            text = edu["text"]
            f.write(f"Utterance {id_nodo} - {speaker}: {text}\n")
            id_nodo += 1
        f.write("\n")
    print(f"Esportato: {nome_file_output}")

# Funzione per caricare il file JSON
def carica_json_da_file(nome_file):
    with open(nome_file, 'r') as file:
        return json.load(file)


# Esegui il programma
if __name__ == "__main__":
    # Chiedi all'utente di inserire il nome del file JSON
    nome_file = "dataset/STAC/train_subindex.json"

    try:
        # Carica i dati dal file JSON
        input_json = carica_json_da_file(nome_file)

        # Creiamo e visualizziamo un grafo per ogni dialogo nel JSON
        for i, dialogo in enumerate(input_json):
            print(f"Creando il grafo per il dialogo {i + 1}...")
            grafo = crea_grafo_da_json([dialogo])  # Passiamo il singolo dialogo come lista
            visualizza_grafo_dag(grafo, i + 1)  # Visualizziamo il grafo del dialogo

            # Esportiamo subito il file di testo relativo al dialogo
            esporta_utterances_singolo(dialogo, i + 1)

        print("Tutti i grafi sono stati creati e i file di testo esportati.")

    except FileNotFoundError:
        print(f"Errore: il file '{nome_file}' non è stato trovato.")
    except json.JSONDecodeError:
        print(f"Errore: il file '{nome_file}' non è un file JSON valido.")
