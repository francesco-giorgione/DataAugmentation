import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATLinkPrediction(nn.Module):
    def __init__(self, embedding_dimension, hidden_channels, num_layers, heads=1):
        super(GATLinkPrediction, self).__init__()

        # Lista dei layer GAT
        self.layers = nn.ModuleList()

        # Primo layer: trasformazione del word embedding iniziale in una rappresentazione nascosta
        self.layers.append(GATConv(embedding_dimension, hidden_channels, heads=heads, concat=True, dropout=0.6))

        # Layer intermedi
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=0.6))

        # Ultimo layer: produce embedding finali, senza concatenazione
        self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6))

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


class LinkPredictionDecoder(nn.Module):
    def __init__(self):
        super(LinkPredictionDecoder, self).__init__()

    def forward(self, z, edge_index):
        # Prende gli embedding di coppie di nodi (archi) e calcola la somiglianza tra di loro
        source, target = edge_index
        z = F.normalize(z, p=2, dim=1, eps=1e-12)
        z_source = z[source]
        z_target = z[target]

        # Calcola il prodotto scalare tra i due embedding per calcolare la somiglianza
        score = (z_source * z_target).sum(dim=1)
        return torch.sigmoid(score)  # Probabilità di link (tra 0 e 1)
