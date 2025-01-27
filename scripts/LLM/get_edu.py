import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.constraints.graph_builder import *


def get_path_dataset(name_dataset):
    if name_dataset == "STAC":
        return '../../dataset/STAC/train_subindex.json'
    elif name_dataset == "MINECRAFT":
        return '../../dataset/MINECRAFT/TRAIN_307_bert.json'
    elif name_dataset == "MOLWENI":
        return '../../dataset/MOLWENI/train.json'
    else: 
        return ''


def get_best_relations(name_dataset):
    if 'STAC' in name_dataset:
        file = '../../info_rel/STAC_rel.txt'
    elif 'MINECRAFT' in name_dataset:
        file = '../../info_rel/MINECRAFT_rel.txt'
    elif 'MOLWENI' in name_dataset:
        file = '../../info_rel/MOLWENI_rel.txt'
    else: 
        return []
    
    best_rel = []

    with open(file, "r") as f:
        for linea in f:
            tipo_relazione = linea.split(":")[0].strip()
            best_rel.append(tipo_relazione)

    dict_best_rel  = {nodo: i for i, nodo in enumerate(reversed(best_rel))}
    return dict_best_rel


def get_edu_from_DAG(name_dataset, graph):
    """
        ritorna l'id della EDU che nel sottodialogo ha la frequenza 
        più alta di archi uscenti e archi entranti
    """



    best_rel = get_best_relations(name_dataset)
    standard_scores = {}
    
    for node in graph.nodes:
        # Prendere gli archi entranti e uscenti
        entranti = list(graph.predecessors(node))
        uscenti = list(graph.successors(node))
        neigh_node = entranti + uscenti
        
        relations = []
        for x in neigh_node:
            if graph.has_edge(x, node):
                relations.append(graph.edges[x, node]["relationship"])
            else:
                relations.append(graph.edges[node, x]["relationship"])

        # print(node, relations)
        sum_pos = sum(best_rel[rel] for rel in relations)
        # print(sum_pos)
        
        # Normalizzare dividendo per il numero di archi
        num_archi = len(neigh_node)
        if num_archi > 0:
            punteggio = sum_pos / num_archi  
        else:
            punteggio = 0
        
        # Salvare il punteggio normalizzato
        standard_scores[node] = punteggio
    
    # Restituire il nodo con il punteggio massimo (cioè posizione più alta in best_rel)
    best_node = max(standard_scores, key=standard_scores.get)
    return best_node, standard_scores


def get_subgraph(target_node, graph):
    if target_node in graph:
        # Ottieni i discendenti (nodi raggiungibili dal target)
        reachable_nodes = nx.descendants(graph, target_node)
        reachable_nodes.add(target_node)  # Includi il nodo target stesso

        # Ottieni gli antenati (nodi che possono raggiungere il target)
        ancestor_nodes = nx.ancestors(graph, target_node)
        ancestor_nodes.add(target_node)  # Includi il nodo target stesso

        # Combina i nodi raggiungibili e gli antenati
        all_relevant_nodes = reachable_nodes.union(ancestor_nodes)

        # Estrai il sottografo
        subgraph = graph.subgraph(all_relevant_nodes)
        
        # Visualizza il sottografo
        # visualizza_grafo_dag(subgraph, 1)

        # Stampa i nodi e gli archi del grafo
        print("Nodi nel grafo:", graph.nodes())
        print("Archi nel grafo:", graph.edges())

        # Stampa i nodi e gli archi del sottografo
        print("Nodi nel sottografo:", subgraph.nodes())
        print("Archi nel sottografo:", subgraph.edges())
    else:
        print(f"Il nodo {target_node} non è presente nel grafo.")

    return subgraph


if __name__ == "__main__":
    name_dataset = "MINECRAFT"
    path_dataset = get_path_dataset(name_dataset)
    dataset_json = carica_json_da_file(path_dataset)
    graph = crea_grafo_da_json([dataset_json[0]])
    visualizza_grafo_dag(graph, 1)

    target_node, standard_scores = get_edu_from_DAG(name_dataset, graph)
    print(f"target node = {target_node}, score = {standard_scores[target_node]}")
    subgraph = get_subgraph(target_node, graph)
