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

        # Calcola la similarità kernel-based
        score = self.rbf_kernel(z_source, z_target)
        return score
    
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

# Classe che utilizza i Multilayer Perceptron per misurare la similarità tra due embeddings
class LinkPredictionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LinkPredictionMLP, self).__init__()

        # Definizione del MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Layer di input
            nn.ReLU(),                            # Funzione di attivazione
            nn.Linear(hidden_dim, 1)              # Layer di output
        )   

        # Inizializzazione dei pesi
        self._init_weights()


    def _init_weights(self):
        # Inizializza i pesi per ogni livello Linear
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):  # Solo per i livelli Linear
                nn.init.kaiming_uniform_(layer.weight)  # Kaiming Initialization per i pesi
                #nn.init.zeros_(layer.bias)            # Inizializza i bias a zero

    def forward(self, z, edge_index):
        # Estrai gli indici sorgente e destinazione degli archi
        source, target = edge_index

        print(source.shape)
        print(target.shape)

        # Normalizza gli embedding
        z = F.normalize(z, p=2, dim=1, eps=1e-12)

        print(z.shape)

        # Recupera gli embedding sorgente e destinazione
        z_source = z[source]
        z_target = z[target]

        print(z_source.shape)
        print(z_target.shape)

        # Concatenazione degli embedding sorgente e destinazione
        z_concat = torch.cat([z_source, z_target], dim=1)

        print(z_concat.shape)
        
        # Calcola la probabilità di link usando l'MLP
        score = self.mlp(z_concat)
        return torch.sigmoid(score).squeeze(dim=1)  # Probabilità di link (tra 0 e 1)