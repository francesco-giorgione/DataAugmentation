import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from statistics import mean
import torch
import torch_geometric
import torch.nn as nn
import json
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scripts.GNN.utils import filter_edge_index, save_models, get_all_rels, plot_loss



# creazione dataset per l'apertura dei DAG
class CustomEduDataset(Dataset):
    def __init__(self, embeddings_path, edges_path):
        # Embeddings dell'edu
        with open(embeddings_path, 'r', encoding='utf-8') as file:
            self.all_edu = json.load(file)

        # Apertura di tutto il DAG
        with open(edges_path, 'r', encoding='utf-8') as file:
            self.all_dag = json.load(file)

        # Numero di nodi massimo
        self.max_nodes = max(len(edu) for edu in self.all_edu)
        self.graphs = []

        for idx, edu in enumerate(self.all_edu):
            # Embedding dei nodi
            embs = [item['embedding'] for item in edu]
            tensor_edu = torch.tensor(embs, dtype=torch.float)

            # Edge index
            edge_index = self.get_edges(idx)

            # Edge positive
            pos_train_edge = [(rel['x'], rel['y']) for rel in self.all_dag[idx]['relations']]
            pos_train_edge = torch.tensor(pos_train_edge, dtype=torch.long)

            """
                Ogni Grafo è rappresentato da un oggetto Data, contenente gli attributi 
                node_emb (con il padding), edge_index e pos_train_edge.
                L'id del grafo non ha subito una variazione dal file json.
            """
            graph = Data(x=tensor_edu, edge_index=edge_index, pos_train_edge=pos_train_edge)
            self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_edges(self, idx):
        dag = self.all_dag[idx]
        """
            Da una lista di coppie, crea due liste (la prima contiene gli archi in entrata e la seconda quelli in uscita)
        """            
        edges = [(rel['x'], rel['y']) for rel in dag['relations']]
        lst1, lst2 = zip(*edges)
        return torch.tensor([lst1, lst2], dtype=torch.long)



"""
    La funzione collate_fn serve per combinare i dati per il DataLoader. Quindi, prende una lista di dati 
    (in questo caso, una lista di oggetti di tipo Data di PyTorch Geometric, ognuno rappresentante un grafo) 
    e la converte in un batch. Il batch è una lista di oggetti, ognuno dei quali rappresenta un grafo con 
    attributi come node_emb, edge_index e pos_train_edge.
"""
def collate_fn(batch):
    return Batch.from_data_list(batch)


# GraphSAGE
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
        super(GNNStack, self).__init__()
        conv_model = torch_geometric.nn.SAGEConv

        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.dropout = dropout
        self.num_layers = num_layers
        self.emb = emb

        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers - 1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # Return node embeddings after post-message passing if specified
        if self.emb:
            return x

        # Else return class probabilities for each node
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def predict(selg, dialogue_json, old_embs, target_node, new_edus_emb, model, link_predictor, threshold=0.5):
        new_embs = old_embs
        new_embs[target_node] = torch.tensor(new_edus_emb, dtype=torch.float32)

        edges = [(rel['x'], rel['y']) for rel in dialogue_json['relations']]
        lst1, lst2 = zip(*edges)
        edge_index = torch.tensor([lst1, lst2], dtype=torch.long)

        pos_train_edge = [(rel['x'], rel['y']) for rel in dialogue_json['relations']]
        pos_train_edge = torch.tensor(pos_train_edge, dtype=torch.long)

        removed_edge_index, filtered_edge_index = filter_edge_index(edge_index, target_node)
        data = Data(x=old_embs, edge_index=filtered_edge_index, pos_train_edge=pos_train_edge)

        # print('Target node', target_node)
        # print('Edge index', edge_index)
        # print('Removed index', removed_edge_index)
        # print('Filtered index', filtered_edge_index)

        node_embs = model(data.x, data.edge_index)
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


# NN di Link Predictor
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                    dropout):
        super(LinkPredictor, self).__init__()

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



def eval_metrics(pos_test_pred, neg_test_pred, threshold = 0.5):
    print(f"pos_test_pred: {pos_test_pred}")
    print(f"neg_test_pred: {neg_test_pred}")

    
    preds = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    labels = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    # threshold = 0.5
    preds_bin = (preds > threshold).float()

    accuracy = accuracy_score(labels.cpu(), preds_bin.cpu())
    precision = precision_score(labels.cpu(), preds_bin.cpu(), zero_division=1)
    recall = recall_score(labels.cpu(), preds_bin.cpu(), zero_division=1)

    return accuracy, precision, recall


def train(model, num_epochs, link_predictor, train_loader, optimizer, path, desc):
    model.train()
    link_predictor.train()
    
    train_losses = []

    with tqdm(total=num_epochs, desc="Training") as progress_bar:
        for epoch in range(num_epochs):
            batch_losses = []

            for i, batch in enumerate(train_loader):
                # print(f"Epoch: {epoch}, batch {i}")
                optimizer.zero_grad()
                
                # Attributi del batch
                node_emb = batch.x  # Embedding dei nodi (N, d)
                # Due liste con archi in entrata e in uscita
                edge_index = batch.edge_index  # Edge index (2, E)
                # (source, target) che indica un arco positivo (realmente presente) nel grafo
                pos_train_edge = batch.pos_train_edge  # Edge positive (B, 2)

                # Passa le embedding e gli archi attraverso il modello
                node_emb = model(node_emb, edge_index)  # (N, d)
                

                
                """
                    link_predictor è un modello che prende gli embedding dei nodi e li usa per predire la
                    probabilità che esista un arco tra due nodi.
                    Gli indici di pos_train_edge (ad esempio, pos_train_edge[:, 0] e pos_train_edge[:, 1]) 
                    vengono utilizzati per ottenere gli embedding dei due nodi che costituiscono un arco positivo. 
                    Quindi, node_emb[pos_train_edge[:, 0]] è l'embedding del nodo di partenza, 
                    e node_emb[pos_train_edge[:, 1]] è l'embedding del nodo di arrivo.
                    La variabile pos_pred è il risultato della previsione (probabilità) che l'arco esista tra i due nodi. 
                    Ogni valore in pos_pred rappresenta la probabilità che un arco esista tra 
                    una coppia di nodi specificata da pos_train_edge.
                """
                pos_pred = link_predictor(node_emb[pos_train_edge[:, 0]], node_emb[pos_train_edge[:, 1]])


                """
                    negative_sampling è una funzione che campiona in modo casuale delle coppie di nodi che 
                    non sono connesse nel grafo (cioè, non sono presenti in edge_index). Questi sono i negativi. 
                    Il numero di coppie negative campionate è pari a num_neg_samples, che corrisponde al numero 
                    di archi positivi, pos_train_edge.shape[0], così da avere lo stesso numero di esempi positivi e negativi.

                    Viene effeuata la trasposizione dell'output di negative_sampling per ottenere la stessa shape di 
                    pos_train_edge. In quanto di default la shape corrisponde a edge_index.
                """
                neg_edge = negative_sampling(edge_index, num_nodes=node_emb.shape[0],
                                            num_neg_samples=pos_train_edge.shape[0], method='dense').T  # (Ne, 2)
                # Previsione dei negativi
                neg_pred = link_predictor(node_emb[neg_edge[:, 0]], node_emb[neg_edge[:, 1]])  # (Ne,)

                """
                    - pos_pred: Le previsioni per gli archi che esistono effettivamente nel grafo. 
                                L'idea è che il modello dovrebbe dare alte probabilità per questi archi.
                    - neg_pred: Le previsioni per gli archi che non esistono nel grafo (campionati negativamente). 
                                Il modello dovrebbe dare basse probabilità per questi archi.
                """

                # print("Pos Pred:", pos_pred.shape)
                # print("Neg Pred:", neg_pred.shape)
                
                """
                    La loss di link prediction verrà calcolata confrontando pos_pred (che dovrebbe essere vicino a 1) 
                    e neg_pred (che dovrebbe essere vicino a 0), spingendo il modello a imparare a fare distinzioni 
                    corrette tra archi esistenti e non esistenti.
                """
                loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
                loss.backward()
                optimizer.step()

                # Aggiungi la loss alla lista dei risultati
                # print(f"Loss: {loss.item()}")
                batch_losses.append(loss.item())
        

            train_losses.append(mean(batch_losses))
            progress_bar.update(1)
            progress_bar.set_postfix({'Loss': loss.item()})

    plot_loss(train_losses, num_epochs, path, desc, isTrain=True)
    return sum(train_losses) / len(train_losses)


def validate(model, link_predictor, loader, threshold, path="", desc="", isTest = False):
    model.eval()
    link_predictor.eval()
    val_loss = []
    pos_pred = []
    neg_pred = []


    with torch.no_grad(), tqdm(total=len(loader), desc="Validation") as progress_bar:   # senza tener traccia dei gradienti
        for i, batch in enumerate(loader):
            # print(f"Batch {i}")
            # Attributi del batch
            node_emb = batch.x  # Embedding dei nodi (N, d)
            # Due liste con archi in entrata e in uscita
            edge_index = batch.edge_index  # Edge index (2, E)
            # (source, target) che indica un arco positivo (realmente presente) nel grafo
            pos_train_edge = batch.pos_train_edge  # Edge positive (B, 2)

            # Passa le embedding e gli archi attraverso il modello
            node_emb = model(node_emb, edge_index)  # (N, d)
            # print(edge_index)

            
            """
                link_predictor è un modello che prende gli embedding dei nodi e li usa per predire la
                probabilità che esista un arco tra due nodi.
                .squeeze() viene utilizzato per assicurarci che le previsioni per ogni arco siano 
                memorizzate come un tensore a 1 dimensione.
            """
            tmp_pos_pred = link_predictor(node_emb[pos_train_edge[:, 0]], node_emb[pos_train_edge[:, 1]])
            pos_pred.append(tmp_pos_pred.squeeze())


            neg_edge = negative_sampling(edge_index, num_nodes=node_emb.shape[0],
                                        num_neg_samples=pos_train_edge.shape[0], method='dense').T  # (Ne, 2)
            # Previsione dei negativi
            tmp_neg_pred = link_predictor(node_emb[neg_edge[:, 0]], node_emb[neg_edge[:, 1]]) 
            neg_pred.append(tmp_neg_pred.squeeze())

            """
                - pos_pred: Le previsioni per gli archi che esistono effettivamente nel grafo. 
                            L'idea è che il modello dovrebbe dare alte probabilità per questi archi.
                - neg_pred: Le previsioni per gli archi che non esistono nel grafo (campionati negativamente). 
                            Il modello dovrebbe dare basse probabilità per questi archi.
            """

            progress_bar.update(1)
            if not isTest:
                loss = -torch.log(tmp_pos_pred + 1e-15).mean() - torch.log(1 - tmp_neg_pred + 1e-15).mean()
                val_loss.append(loss.item())
                progress_bar.set_postfix({'Loss': loss.item()})


    if not isTest:
        loss = sum(val_loss) / len(val_loss)
        plot_loss(val_loss, len(test_loader), path, desc)

    pos_pred = torch.cat(pos_pred, dim=0)
    neg_pred = torch.cat(neg_pred, dim=0)
    accuracy, precision, recall = eval_metrics(pos_pred, neg_pred, threshold=threshold)
    print(f"Accuracy {accuracy}, Precision {precision}, Recall {recall}")
    return {"accuracy" : accuracy, "precision" : precision, "recall" : recall}


def load_models(path, num_layers):
    model = GNNStack(input_dim=768, hidden_dim=768, output_dim=384, num_layers=num_layers, dropout=0.3)                          # MiniLM 384 - MPNet 768
    link_predictor = LinkPredictor(in_channels=384, hidden_channels=192, out_channels=1, num_layers=num_layers, dropout=0.3) 

    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)  
    model.load_state_dict(checkpoint['model_state_dict'])
    link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])

    print(f"Modelli caricati da {path}")
    return model, link_predictor


if __name__ == "__main__":
    """
        Dataset che contiene tutte le edu per in dato DAG.
        ID_DAG = ID_ELEMENTO_DATASET
    """
    train_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/STAC_training_embeddings.json',
        edges_path='../../dataset/STAC/train_subindex.json'
    )
    val_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/STAC_val_embeddings.json',
        edges_path='../../dataset/STAC/dev.json'
    )
    test_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/STAC_testing_embeddings.json',
        edges_path='../../dataset/STAC/test_subindex.json'
    )

    """ train_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/MINECRAFT_training_embeddings.json',
        edges_path='../../dataset/MINECRAFT/TRAIN_307_bert.json'
    )
    val_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/MINECRAFT_val_embeddings.json',
        edges_path='../../dataset/MINECRAFT/VAL_all.json'
    )
    test_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/MINECRAFT_testing133_embeddings.json',
        edges_path='../../dataset/MINECRAFT/TEST_133.json'
    )
    """

    """ train_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/MOLWENI_training_embeddings.json',
        edges_path='../../dataset/MOLWENI/train.json'
    )
    val_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/MOLWENI_val_embeddings.json',
        edges_path='../../dataset/MOLWENI/dev.json'
    )
    test_dag = CustomEduDataset(
        embeddings_path='../../embeddings/MPNet/MOLWENI_testing_embeddings.json',
        edges_path='../../dataset/MOLWENI/test.json'
    ) """
    

    """
        Il DataLoader gestisce la divisione del dataset in batch, per migliorare l'efficienza 
        e la precisione nella fase training. Il parametro shuffle se impostato a True rimescola 
        i dati in ogni epoca. Tale funzionalità è utile per il training, ma non per il test.
    """
    batch_size = 32
    train_loader = DataLoader(train_dag, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dag, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dag, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    model = GNNStack(input_dim=768, hidden_dim=768, output_dim=384, num_layers=3, dropout=0.3)                          # MiniLM 384 - MPNet 768
    link_predictor = LinkPredictor(in_channels=384, hidden_channels=192, out_channels=1, num_layers=3, dropout=0.3)     # MiniLM 128 - MPNet 384

    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=0.0001)
    model, link_predictor = load_models("pretrain_model_GS/Minecraft_pretrained_models.pth", num_layers = 3)
    # train(model, 100, link_predictor, train_loader, optimizer, path="plot_loss/GS_STAC_train.png", desc= "STAC Training Loss")
    
    # save_models(model, link_predictor, 'pretrain_model_GS/STAC_pretrained_models.pth')

    validate(model, link_predictor, val_loader, threshold=0.5, path="plot_loss/GS_MINECRAFT_val.png", desc= "STAC Validation Loss")
    # validate(model, link_predictor, test_loader, threshold=0.5, isTest=True)
    # validate(val_loader, model, link_predictor, threshold=0.5)
