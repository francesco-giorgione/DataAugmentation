import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.optim as optim

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


# Classe che utilizza il dot product come misura di similarità tra due embeddings
class LinkPredictionDecoder(nn.Module):
    def __init__(self):
        super(LinkPredictionDecoder, self).__init__()

    def forward(self, z, edge_index):
        # Estrai gli indici sorgente e destinazione degli archi
        source, target = edge_index
        # Normalizza gli embedding
        z = F.normalize(z, p=2, dim=1, eps=1e-12)
        # Recupera gli embedding sorgente e destinazione
        z_source = z[source]
        z_target = z[target]

        # Calcola il prodotto scalare tra i due embedding per calcolare la somiglianza
        score = (z_source * z_target).sum(dim=1)
        return torch.sigmoid(score)  # Probabilità di link
    
# Classe che utilizza la Kernel-based Similarity come misura di similarità tra due embeddings
class LinkPredictionDecoderKernel(nn.Module):
    def __init__(self, sigma):
        super(LinkPredictionDecoderKernel, self).__init__()
        self.sigma = sigma  # Parametro del kernel RBF

    def rbf_kernel(self, z_source, z_target):
        # Calcolo della distanza quadrata tra z_source e z_target
        dist_squared = torch.sum((z_source - z_target) ** 2, dim=1)  # Norm 2
        # Applica il kernel RBF
        return torch.exp(-dist_squared / (2 * self.sigma ** 2))

    def forward(self, z, edge_index):
        # Estrai gli indici sorgente e destinazione degli archi
        source, target = edge_index
        # Normalizza gli embedding
        z = F.normalize(z, p=2, dim=1, eps=1e-12)
        # Recupera gli embedding sorgente e destinazione
        z_source = z[source]
        z_target = z[target]

        # Calcola la similarità con il dot product
        # score = (z_source * z_target).sum(dim=1)

        # Calcola la similarità kernel-based
        score = self.rbf_kernel(z_source, z_target)
        return score


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