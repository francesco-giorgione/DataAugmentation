import json
import torch
from torch_geometric.utils import negative_sampling
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.data import Data
from GAT import GATLinkPrediction, LinkPredictionDecoder, LinkPredictionDecoderKernel, LinkPredictorMLP
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset


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

            # Padding degli embeddings
            padding_size = self.max_nodes - len(tensor_edu)
            padded_emb = torch.cat([tensor_edu, torch.zeros(padding_size, tensor_edu.size(1))], dim=0)

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
            graph = Data(x=padded_emb, edge_index=edge_index, pos_train_edge=pos_train_edge)
            self.graphs.append(graph)

        print('self.grpahs[0]', self.graphs[0])

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
    threshold = 0.6
    preds_bin = (preds > threshold).float()

    accuracy = accuracy_score(labels.cpu(), preds_bin.cpu())
    precision = precision_score(labels.cpu(), preds_bin.cpu())
    recall = recall_score(labels.cpu(), preds_bin.cpu())

    return accuracy, precision, recall


def test_worker(model, predictor, emb, edge_index, pos_test_edge):
    """
    Evaluates model on positive and negative test edges
    1. Computes the updated node embeddings given the edge index (i.e. the message passing edges)
    2. Computes predictions on the positive and negative edges
    3. Calculates hits @ k given predictions using the ogb evaluator
    """
    model.eval()
    predictor.eval()

    # CODICE AGGIUNTO IN SCRIPT INIZIALE
    pos_test_edge = pos_test_edge.T
    neg_test_edge = negative_sampling(edge_index, num_nodes=emb.shape[0], num_neg_samples=pos_test_edge.size(0), method='dense').to(emb.device)
    neg_test_edge = neg_test_edge.T

    data = Data(x=emb, edge_index=edge_index)   # (N, d)
    node_emb = model(data)

    # CODICE RIMOSSO DA SCRIPT INIZIALE
    # pos_test_edge = split_edge['test']['edge'].to(emb.device)
    # neg_test_edge = split_edge['test']['edge_neg'].to(emb.device)

    pos_test_preds = []
    # batch_size=pos_train_edge.shape[0] affinché il singolo aggiornamento consideri tutti gli archi
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size=pos_test_edge.size(0)):
        edge = pos_test_edge[perm].t()
        # pos_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
        pos_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    # batch_size=neg_train_edge.shape[0] affinché il singolo aggiornamento consideri tutti gli archi
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size=neg_test_edge.size(0)):
        edge = neg_test_edge[perm].t()
        # neg_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
        neg_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    print(f'pos test pred {pos_test_pred}\nneg test pred {neg_test_pred}')
    return eval_metrics(pos_test_pred, neg_test_pred)


def train_worker(model, link_predictor, emb, edge_index, pos_train_edge, batch_size, optimizer):
    """
    Runs offline training for model, link_predictor and node embeddings given the message
    edges and supervision edges.
    1. Updates node embeddings given the edge index (i.e. the message passing edges)
    2. Computes predictions on the positive supervision edges
    3. Computes predictions on the negative supervision edges (which are sampled)
    4. Computes the loss on the positive and negative edges and updates parameters
    """

    model.train()
    link_predictor.train()

    train_losses = []

    # print('prima della trasposizione', pos_train_edge)
    pos_train_edge = pos_train_edge.T
    # print('dopo la trasposizione', pos_train_edge)
    # print(pos_train_edge.shape)

    # batch_size=pos_train_edge.shape[0] affinché il singolo aggiornamento consideri tutti gli archi
    for edge_id in DataLoader(range(pos_train_edge.shape[0]), batch_size=pos_train_edge.shape[0], shuffle=True):
        # optimizer.zero_grad()

        data = Data(x=emb, edge_index=edge_index)   # (N, d)
        node_emb = model(data)

        pos_edge = pos_train_edge[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        neg_edge = negative_sampling(edge_index, num_nodes=emb.shape[0],
                                     num_neg_samples=edge_id.shape[0], method='dense')  # (Ne,2)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (Ne,)

        loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()

        # COMMENTATO perché voglio che l'aggiornamento sia fatto sul mini-batch di dialoghi, non sul singolo dialogo
        # loss.backward()
        # optimizer.step()

        train_losses.append(loss)

    # Codice commentato perché l'intero dialogo è processato tutto insieme: ho un unico valore di loss
    # res = sum(train_losses) / len(train_losses)

    # Restituisce la loss sul singolo dialogo
    return train_losses[0]


def test(dataset_filename, embs_filename, model, link_predictor, batch_size=1000):
    total_accuracy, total_precision, total_recall = 0, 0, 0

    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    n = len(all_dialogues)
    num_samples = min(batch_size, n)
    sampled_dialogues = random.sample(range(n), num_samples)

    for dialogue_index in sampled_dialogues:
        embs = torch.tensor([item['embedding'] for item in all_embs[dialogue_index]], dtype=torch.float)
        edge_index = super_new_get_edges(all_dialogues, dialogue_index)

        print(f'[Testing] Sampled dialogue {dialogue_index}')
        accuracy, precision, recall = test_worker(model, link_predictor, embs, edge_index, edge_index)

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall

    accuracy = total_accuracy / num_samples
    precision = total_precision / num_samples
    recall = total_recall / num_samples
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')


def old_train(dataset_filename, embs_filename, num_epochs=100, batch_size=50, learning_rate=0.001, model=None, link_predictor=None):
    if model is None:
        model = GATLinkPrediction(embedding_dimension=768, hidden_channels=256, num_layers=2, heads=16)

    if link_predictor is None:
        link_predictor = LinkPredictorMLP(in_channels=256, hidden_channels=256, out_channels=1, num_layers=4, dropout=0.5)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=learning_rate)

    n = n_dialogues(dataset_filename)
    num_samples = min(batch_size, n)

    for epoch in range(num_epochs):
        batch_losses = []

        for batch_index, batch_dialogues in enumerate(DataLoader(range(n), batch_size=batch_size, shuffle=True, num_workers=4), start=1):
            dialogue_losses = []
            batch_dialogues = batch_dialogues.tolist()
            print('Sampled dialogues:', batch_dialogues)

            for dialogue_index in batch_dialogues:
                # print(f'Sampled dialogue {dialogue_index}')
                embs = get_embs(embs_filename, dialogue_index)
                print('embs', embs)
                edge_index = get_edges(dataset_filename, dialogue_index)
                dialogue_losses.append(train_worker(model, link_predictor, embs, edge_index, edge_index, batch_size, optimizer))

            batch_loss = torch.stack(dialogue_losses).mean()
            batch_losses.append(batch_loss)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_index} -> Batch training Loss: {batch_loss:.4f}")

        print(f'Num batch in epoch {epoch}: {batch_index}')
        epoch_loss = torch.stack(batch_losses).mean()
        print(f"Epoch {epoch+1}/{num_epochs}, Training epoch loss: {epoch_loss:.4f}")

        save_models(model, link_predictor, file_path)

    return model, link_predictor


def old_dataset_train(train_dataset, num_epochs=100, batch_size=50, learning_rate=0.001, model=None, link_predictor=None):
    if model is None:
        model = GATLinkPrediction(embedding_dimension=768, hidden_channels=256, num_layers=2, heads=16)

    if link_predictor is None:
        link_predictor = LinkPredictorMLP(in_channels=256, hidden_channels=256, out_channels=1, num_layers=4, dropout=0.5)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=learning_rate)

    # n = n_dialogues(dataset_filename)

    for epoch in range(num_epochs):
        batch_losses = []

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        for batch_index, (indices, batch_dialogues) in enumerate(zip(train_loader.batch_sampler, train_loader), start=1):
            dialogue_losses = []
            batch_dialogues = batch_dialogues.to_data_list()
            print('Sampled dialogues:', indices)
            # print('Batch dialogues:', batch_dialogues)

            for dialogue in batch_dialogues:
                embs = dialogue.x
                edge_index = dialogue.edge_index
                print('edge_index:', edge_index)
                dialogue_losses.append(train_worker(model, link_predictor, embs, edge_index, edge_index, batch_size, optimizer))

            batch_loss = torch.stack(dialogue_losses).mean()
            batch_losses.append(batch_loss)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_index} -> Batch training Loss: {batch_loss:.4f}")

        print(f'Num batch in epoch {epoch}: {batch_index}')
        epoch_loss = torch.stack(batch_losses).mean()
        print(f"Epoch {epoch+1}/{num_epochs}, Training epoch loss: {epoch_loss:.4f}")

        # save_models(model, link_predictor, file_path)

    return model, link_predictor


def train(dataset_filename, embs_filename, num_epochs=100, batch_size=50, learning_rate=0.001, model=None, link_predictor=None):
    if model is None:
        model = GATLinkPrediction(embedding_dimension=768, hidden_channels=256, num_layers=2, heads=16)

    if link_predictor is None:
        link_predictor = LinkPredictorMLP(in_channels=256, hidden_channels=256, out_channels=1, num_layers=4, dropout=0.5)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=learning_rate)

    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    n = len(all_dialogues)

    for epoch in range(num_epochs):
        batch_losses = []

        for batch_index, batch_dialogues in enumerate(DataLoader(range(n), batch_size=batch_size, shuffle=True, num_workers=4), start=1):
            dialogue_losses = []
            batch_dialogues = batch_dialogues.tolist()
            print('Sampled dialogues:', batch_dialogues)

            for dialogue_index in batch_dialogues:
                embs = torch.tensor([item['embedding'] for item in all_embs[dialogue_index]], dtype=torch.float)
                edge_index = super_new_get_edges(all_dialogues, dialogue_index)

                dialogue_losses.append(train_worker(model, link_predictor, embs, edge_index, edge_index, batch_size, optimizer))

            batch_loss = torch.stack(dialogue_losses).mean()
            batch_losses.append(batch_loss)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_index} -> Batch training Loss: {batch_loss:.4f}")

        print(f'Num batch in epoch {epoch}: {batch_index}')
        epoch_loss = torch.stack(batch_losses).mean()
        print(f"Epoch {epoch+1}/{num_epochs}, Training epoch loss: {epoch_loss:.4f}")

        save_models(model, link_predictor, file_path)

    return model, link_predictor



if __name__ == '__main__':
    file_path = 'pretrained_models_MINECRAFT.pth'
    trained_model, trained_link_predictor = load_models(file_path)

    trained_model, trained_link_predictor = train('../../dataset/MOLWENI/train.json',
        '../../embeddings/MPNet/MOLWENI_training_embeddings.json', num_epochs=10, model=trained_model, link_predictor=trained_link_predictor)

    test('../../dataset/MINECRAFT/TEST_133.json', '../../embeddings/MPNet/MINECRAFT_testing133_embeddings.json', trained_model, trained_link_predictor)




