import json
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.data import Data
from GAT import GATLinkPrediction, LinkPredictorMLP
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset
import networkx as nx


def load_data(dataset_filename):
    with open(dataset_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def n_dialogues(dataset_filename):
    with open(dataset_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return len(data)


def save_models(model, link_predictor, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'link_predictor_state_dict': link_predictor.state_dict()
    }, path)
    print(f"Modelli salvati in {path}")


def load_models(path):
    model = GATLinkPrediction(embedding_dimension=768, hidden_channels=256, num_layers=2, heads=16)
    link_predictor = LinkPredictorMLP(in_channels=256, hidden_channels=256, out_channels=1, num_layers=4, dropout=0.5)

    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # Usa 'cuda' se hai una GPU
    model.load_state_dict(checkpoint['model_state_dict'])
    link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])

    print(f"Modelli caricati da {path}")
    return model, link_predictor


def get_embs(embs_filename, dialogue_index):
    with open(embs_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)[dialogue_index]

    embs = [item['embedding'] for item in data]
    return torch.tensor(embs, dtype=torch.float)


def get_edges(dataset_filename, dialogue_index):
    with open(dataset_filename, 'r', encoding='utf-8') as file:
        tmp = [(rel['x'], rel['y']) for rel in json.load(file)[dialogue_index]['relations']]

    # print(tmp)
    lst1, lst2 = zip(*tmp)
    lst1 = list(lst1)
    lst2 = list(lst2)

    return torch.tensor([lst1,  # nodi di origine
                         lst2],  # nodi di destinazione
                        dtype=torch.long)


def super_new_get_edges(all_dialogues, dialogue_index):
    tmp = [(rel['x'], rel['y']) for rel in all_dialogues[dialogue_index]['relations']]

    # print(tmp)
    lst1, lst2 = zip(*tmp)
    lst1 = list(lst1)
    lst2 = list(lst2)

    return torch.tensor([lst1,  # nodi di origine
                         lst2],  # nodi di destinazione
                        dtype=torch.long)


def eval_metrics(pos_test_pred, neg_test_pred):
    preds = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    labels = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    threshold = 0.5
    preds_bin = (preds > threshold).float()

    accuracy = accuracy_score(labels.cpu(), preds_bin.cpu())
    precision = precision_score(labels.cpu(), preds_bin.cpu())
    recall = recall_score(labels.cpu(), preds_bin.cpu())

    return accuracy, precision, recall


def filter_edge_index(edge_index, target_node):
    mask = (edge_index[0] != target_node) & (edge_index[1] != target_node)
    removed = ~mask
    filtered_index = edge_index[:, mask]
    removed_index = edge_index[:, removed]

    return removed_index, filtered_index


def create_graph(dialogue):
    G = nx.DiGraph()
    id_nodo = 0

    for edu in dialogue["edus"]:
        G.add_node(id_nodo, text = edu["text"], speaker = edu["speaker"])
        id_nodo += 1

    for relazione in dialogue["relations"]:
        x = relazione["x"]
        y = relazione["y"]
        relazione_tipo = relazione["type"]
        G.add_edge(x, y, relationship=relazione_tipo)

    return G


def get_best_relations(file_path):
    best_rel = []

    with open(file_path, "r") as f:
        for linea in f:
            tipo_relazione = linea.split(":")[0].strip()
            best_rel.append(tipo_relazione)

    dict_best_rel  = {nodo: i for i, nodo in enumerate(reversed(best_rel))}
    return dict_best_rel


def get_target_node(name_dataset, graph):
    """
        Restituisce l'id della EDU che nel sottodialogo ha la frequenza
        più alta di archi uscenti e archi entranti
    """

    if 'STAC' in name_dataset:
        file_path = '../../info_rel/STAC_rel.txt'
    elif 'MOLWENI' in name_dataset:
        file_path = '../../info_rel/MOLWENI_rel.txt'
    elif 'MINECRAFT' in name_dataset:
        file_path = '../../info_rel/MINECRAFT_rel.txt'
    else:
        return None

    best_rel = get_best_relations(file_path)
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


def get_all_rels(dialogue_json, target_node):
    in_rels, out_rels = [], []

    for rel in dialogue_json['relations']:
        if rel['y'] == target_node:
            in_rels.append(rel['x'])
        elif rel['x'] == target_node:
            out_rels.append(rel['y'])

    return in_rels, out_rels
