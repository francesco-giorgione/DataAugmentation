import json
import torch
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.data import Data
from GAT import GATLinkPrediction, LinkPredictionDecoder, LinkPredictionDecoderKernel, LinkPredictorMLP
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score


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


def eval_metrics(pos_test_pred, neg_test_pred):
    preds = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    labels = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    threshold = 0.5
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


def test(dataset_filename, embs_filename, model, link_predictor, batch_size=150):
    # model = GATLinkPrediction(embedding_dimension=768, hidden_channels=256, num_layers=2, heads=16)
    # link_predictor = LinkPredictorMLP(in_channels=256, hidden_channels=256, out_channels=1, num_layers=4, dropout=0.5)

    n = n_dialogues(dataset_filename)
    num_samples = min(batch_size, n)
    sampled_dialogues = random.sample(range(n), num_samples)

    total_accuracy, total_precision, total_recall = 0, 0, 0

    for dialogue_index in sampled_dialogues:
        embs = get_embs(embs_filename, dialogue_index)
        edge_index = get_edges(dataset_filename, dialogue_index)

        print(f'[Testing] Sampled dialogue {dialogue_index}')
        accuracy, precision, recall = test_worker(model, link_predictor, embs, edge_index, edge_index)

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall

    accuracy = total_accuracy / num_samples
    precision = total_precision / num_samples
    recall = total_recall / num_samples
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')


def train(dataset_filename, embs_filename, num_epochs=100, batch_size=50, learning_rate=0.001, model=None, link_predictor=None):
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


if __name__ == '__main__':
    file_path = 'pretrained_models.pth'
    trained_model, trained_link_predictor = load_models(file_path)

    trained_model, trained_link_predictor = train('../../dataset/STAC/train_subindex.json',
        '../../embeddings/MPNet/STAC_training_embeddings.json', num_epochs=3, model=trained_model, link_predictor=trained_link_predictor)

    test('../../dataset/STAC/test_subindex.json', '../../embeddings/MPNet/STAC_testing_embeddings.json', trained_model, trained_link_predictor)

