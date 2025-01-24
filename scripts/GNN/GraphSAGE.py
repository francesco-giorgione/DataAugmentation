import os
from statistics import mean
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import torch
import torch_geometric
import torch.nn as nn
import json
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import Evaluator
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import create_graph, filter_edge_index, get_target_node, save_models



# creazione dataset per l'apertura dei DAG
class CustomEduDataset(Dataset):
    def __init__(self, embeddings_path, edges_path, nome_dataset = "", isValidation = False):
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

            # Padding degli embeddings
            padding_size = self.max_nodes - len(tensor_edu)
            padded_emb = torch.cat([tensor_edu, torch.zeros(padding_size, tensor_edu.size(1))], dim=0)

            # Edge index
            edge_index = self.get_edges(idx)

            # Edge positive
            pos_train_edge = [(rel['x'], rel['y']) for rel in self.all_dag[idx]['relations']]
            pos_train_edge = torch.tensor(pos_train_edge, dtype=torch.long)
            removed_edges = torch.empty((2, 0), dtype=edge_index.dtype)

            if isValidation:
                graph = create_graph(self.all_dag[idx])
                target_node = get_target_node(nome_dataset, graph)
                removed_edges, edge_index = filter_edge_index(edge_index, target_node)
                _, pos_train_edge = filter_edge_index(pos_train_edge, target_node)

            """
                Ogni Grafo è rappresentato da un oggetto Data, contenente gli attributi 
                node_emb (con il padding), edge_index e pos_train_edge.
                L'id del grafo non ha subito una variazione dal file json.
            """
            graph = Data(x=tensor_edu, edge_index=edge_index, pos_train_edge=pos_train_edge, removed_edges=removed_edges)
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


def plot_loss(loss_history, num_epochs, path, desc="Loss"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), loss_history, color='red', label='Training Loss')
    plt.title(desc)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    
    # Gestisci le etichette sull'asse x
    if num_epochs > 30:
        step = max(1, num_epochs // 5)  # Mostra circa 10 etichette
        plt.xticks(range(1, num_epochs + 1, step))
    else:
        plt.xticks(range(1, num_epochs + 1))
    
    # Calcolo del margine aggiuntivo (10% sopra e sotto i valori min/max)
    min_loss = min(loss_history)
    max_loss = max(loss_history)
    margin = (max_loss - min_loss) * 0.1 + 2
    plt.ylim(min_loss - margin, max_loss + margin)
    
    # Formatta i numeri dell'asse y con due cifre decimali
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    
    plt.margins(0.05)
    plt.tight_layout()
    plt.savefig(path)


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

    plot_loss(train_losses, num_epochs, path, desc)
    return sum(train_losses) / len(train_losses)


def test(model, link_predictor, test_loader, threshold, path, desc):
    model.eval()
    link_predictor.eval()
    test_losses = []
    pos_pred = []
    neg_pred = []


    with torch.no_grad(), tqdm(total=len(test_loader), desc="Validation") as progress_bar:   # senza tener traccia dei gradienti
        for i, batch in enumerate(test_loader):
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

            loss = -torch.log(tmp_pos_pred + 1e-15).mean() - torch.log(1 - tmp_neg_pred + 1e-15).mean()

            # Aggiungi la loss alla lista dei risultati
            # print(f"Loss: {loss.item()}")
            test_losses.append(loss.item())
            progress_bar.update(1)
            progress_bar.set_postfix({'Loss': loss.item()})


    plot_loss(test_losses, len(test_loader), path, desc)
    pos_pred = torch.cat(pos_pred, dim=0)
    neg_pred = torch.cat(neg_pred, dim=0)
    accuracy, precision, recall = eval_metrics(pos_pred, neg_pred, threshold=threshold)
    loss = sum(test_losses) / len(test_losses)
    print(f"Accuracy {accuracy}, Precision {precision}, Recall {recall}, Loss {loss}")
    return {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "loss" : loss}


def load_models(path, num_layers):
    model = GNNStack(input_dim=768, hidden_dim=768, output_dim=384, num_layers=3, dropout=0.3)                          # MiniLM 384 - MPNet 768
    link_predictor = LinkPredictor(in_channels=384, hidden_channels=192, out_channels=1, num_layers=3, dropout=0.3) 

    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)  
    model.load_state_dict(checkpoint['model_state_dict'])
    link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])

    print(f"Modelli caricati da {path}")
    return model, link_predictor


def validate(val_loader, model, link_predictor, threshold=0.5):
    model.eval()
    link_predictor.eval()
    total_predictions = 0
    total_correct_predictions = 0

    with torch.no_grad(), tqdm(total=len(val_loader), desc="Validating") as progress_bar:
        for i, batch in enumerate(val_loader):
            curr_correct_prediction_counter = 0
            node_emb = batch.x  # Embedding dei nodi (N, d)
            edge_index = batch.edge_index.squeeze(1)  # Edge index (2, E)
            removed_edges = batch.removed_edges  # Archi rimossi per validazione

            # Salta batch vuoti
            if node_emb.numel() == 0 or edge_index.numel() == 0:
                progress_bar.update(1)
                continue

            node_emb = model(node_emb, edge_index)

            for i in range(removed_edges.shape[1]):
                emb_1 = node_emb[removed_edges[0, i]]
                emb_2 = node_emb[removed_edges[1, i]]
                # print(emb_1, emb_2)

                prob = link_predictor(emb_1, emb_2)
                # print(prob)

                if prob >= threshold:
                    curr_correct_prediction_counter += 1

            progress_bar.update(1)
            progress_bar.set_postfix({
                'Correct Predictions': f"{curr_correct_prediction_counter}/{removed_edges.size(1)}"
            })

    # Calcola l'accuratezza complessiva
    accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print(f'Totale predizioni corrette: {total_correct_predictions}/{total_predictions} '
          f'({accuracy * 100:.2f}%)')
    
    return accuracy



if __name__ == "__main__":
    """
        Dataset che contiene tutte le edu per in dato DAG.
        ID_DAG = ID_ELEMENTO_DATASET
    """
    train_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/STAC_training_embeddings.json',
        edges_path='dataset/STAC/train_subindex.json'
    )
    val_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/STAC_val_embeddings.json',
        edges_path='dataset/STAC/dev.json'
    )
    test_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/STAC_testing_embeddings.json',
        edges_path='dataset/STAC/test_subindex.json'
    )

    """ train_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/MINECRAFT_training_embeddings.json',
        edges_path='dataset/MINECRAFT/TRAIN_307_bert.json'
    )
    val_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/MINECRAFT_val_embeddings.json',
        edges_path='dataset/MINECRAFT/VAL_all.json'
    )
    test_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/MINECRAFT_testing133_embeddings.json',
        edges_path='dataset/MINECRAFT/TEST_133.json'
    )
    """

    """ train_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/MOLWENI_training_embeddings.json',
        edges_path='dataset/MOLWENI/train.json'
    )
    val_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/MOLWENI_val_embeddings.json',
        edges_path='dataset/MOLWENI/dev.json'
    )
    test_dag = CustomEduDataset(
        embeddings_path='embeddings/MPNet/MOLWENI_testing_embeddings.json',
        edges_path='dataset/MOLWENI/test.json'
    ) """
    

    """
        Il DataLoader gestisce la divisione del dataset in batch, per migliorare l'efficienza 
        e la precisione nella fase training. Il parametro shuffle se impostato a True rimescola 
        i dati in ogni epoca. Tale funzionalità è utile per il training, ma non per il test.
    """
    batch_size = 32
    train_loader = DataLoader(train_dag, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dag, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dag, batch_size=32, shuffle=False, collate_fn=collate_fn)


    model = GNNStack(input_dim=768, hidden_dim=768, output_dim=384, num_layers=4, dropout=0.3)                          # MiniLM 384 - MPNet 768
    link_predictor = LinkPredictor(in_channels=384, hidden_channels=192, out_channels=1, num_layers=4, dropout=0.3)     # MiniLM 128 - MPNet 384

    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=0.0001)
    #model, link_predictor = load_models("pretrain_model_GS/Molweni_pretrained_models_1.pth", num_layers = 3)
    train(model, 100, link_predictor, train_loader, optimizer, path="plot_loss/GS_STAC_train.png", desc= "STAC Training Loss")
    
    save_models(model, link_predictor, 'pretrain_model_GS/STAC_pretrained_models.pth')

    test(model, link_predictor, val_loader, threshold=0.5, path="plot_loss/GS_STAC_val.png", desc= "STAC Validation Loss")
    # validate(val_loader, model, link_predictor, threshold=0.5)
