import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import sys
import os
from torch_geometric.data import Data
from scripts.GNN.utils import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch.optim as optim

class GATLinkPrediction(nn.Module):
    def __init__(self, embedding_dimension, hidden_channels, num_layers, dropout = 0.6, heads=1):
        super(GATLinkPrediction, self).__init__()

        # Lista dei layer GAT
        self.layers = nn.ModuleList()

        # Primo layer: trasformazione del word embedding iniziale in una rappresentazione nascosta
        self.layers.append(GATConv(embedding_dimension, hidden_channels, heads=heads, concat=True, dropout=dropout))

        # Layer intermedi
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))

        # Ultimo layer: produce embedding finali, senza concatenazione
        self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, data):
        # 'data.x' è il tensore contenente i word embeddings dei nodi (es. parole)
        # 'data.edge_index' è il tensore che contiene gli archi (connettività del grafo)
        x, edge_index = data.x, data.edge_index

        # Passaggio attraverso i layer GAT
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.elu(x)  # Funzione di attivazione (ELU)

        # Ultimo layer: esegui attenzione e aggregazione finale
        x = self.layers[-1](x, edge_index)

        return x  # Gli embedding finali dei nodi (parole)


    def predict(self, dialogue_json, old_embs, target_node, new_edus_emb, model, link_predictor, threshold=0.5):
        new_embs = old_embs
        new_embs[target_node] = torch.tensor(new_edus_emb, dtype=torch.float32)

        edge_index = super_new_get_edges([dialogue_json], 0)
        removed_edge_index, filtered_edge_index = filter_edge_index(edge_index, target_node)

        # print('Target node', target_node)
        # print('Edge index', edge_index)
        # print('Removed index', removed_edge_index)
        # print('Filtered index', filtered_edge_index)

        data = Data(x=new_embs, edge_index=filtered_edge_index)   # (N, d)
        node_embs = model(data)
        in_rels, out_rels = get_all_rels(dialogue_json, target_node)

        to_predict_edges = []
        print(f'In_rels: {in_rels}, Out_rels: {out_rels}')

        for rel in in_rels:
            emb_src = node_embs[rel]
            emb_dst = node_embs[target_node]
            to_predict_edges.append((emb_src, emb_dst))

        for rel in out_rels:
            emb_src = node_embs[target_node]
            emb_dst = node_embs[rel]
            to_predict_edges.append((emb_src, emb_dst))

        predicted_probs_for_edges = []
        for edge in to_predict_edges:
            edge_prob = link_predictor(edge[0], edge[1]).item()
            print('Predicted prob:', edge_prob)
            predicted_probs_for_edges.append(edge_prob)

        return predicted_probs_for_edges


class LinkPredictorMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictorMLP, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)