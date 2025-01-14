import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from GAT import GATLinkPrediction, LinkPredictionDecoder


def test_json(embs_filename, dialogue_index):
    with open(embs_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)[dialogue_index]

    for item in data:
        print(item)
        print('\n\n\n\n')


def get_embs(embs_filename, dialogue_index):
    with open(embs_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)[dialogue_index]

    embs = [item['embedding'] for item in data]
    return torch.tensor(embs, dtype=torch.float)


def get_edges(dataset_filename, dialogue_index):
    with open(dataset_filename, 'r', encoding='utf-8') as file:
        tmp = [(rel['x'], rel['y']) for rel in json.load(file)[dialogue_index]['relations']]

    print(tmp)
    lst1, lst2 = zip(*tmp)
    lst1 = list(lst1)
    lst2 = list(lst2)

    return torch.tensor([lst1,  # nodi di origine
                      lst2],  # nodi di destinazione
                     dtype=torch.long)


def get_all_edges(embs):
    n = embs.size()[0]

    origin, dest = [], []

    for i in range(n):
        for j in range(n):
            if i != j:
                origin.append(i)
                dest.append(j)

    return torch.tensor([origin, dest], dtype=torch.long)


def get_labels(all_edges, true_edges):
    labels = torch.zeros(all_edges.size(1), dtype=torch.float)
    true_edges_set = set(tuple(true_edges[:, i].tolist()) for i in range(true_edges.size(1)))

    for i in range(all_edges.size(1)):
        # Verifica se l'arco in all_edges esiste in true_edges_set
        edge_tuple = (all_edges[0, i].item(), all_edges[1, i].item())
        if edge_tuple in true_edges_set:
            labels[i] = 1.0  # L'arco esiste veramente

    return labels


def loss_fn(pred, labels):
    return F.binary_cross_entropy(pred, labels)


def get_metrics(predictions, labels):
    TP = torch.sum((predictions == 1) & (labels == 1)).item()
    FP = torch.sum((predictions == 1) & (labels == 0)).item()
    TN = torch.sum((predictions == 0) & (labels == 0)).item()
    FN = torch.sum((predictions == 0) & (labels == 1)).item()

    accuracy = (TP + TN) / len(predictions)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return accuracy, precision, recall


def print_results(probabilities, predictions, labels):
    print('Probabilities:', probabilities)
    print('Predictions:', predictions)
    print('Labels:', labels)

    loss = loss_fn(probabilities, labels)
    accuracy, precision, recall = get_metrics(predictions, labels)
    print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')


def test(model, embs, true_edges):
    all_edges = get_all_edges(embs)
    labels = get_labels(all_edges, true_edges)

    model.eval()
    data = Data(x=embs, edge_index=true_edges)

    with torch.no_grad():
        updated_embs = model(data)

    print(f'Embedding node 0:', updated_embs[0])
    print(f'Embedding node 1:', updated_embs[1])

    decoder = LinkPredictionDecoder()
    probabilities = decoder(updated_embs, all_edges)
    predictions = (probabilities >= 0.6).float()

    print_results(probabilities, predictions, labels)


def train(embs, true_edges, num_epochs=100, learning_rate=0.01):
    all_edges = get_all_edges(embs)
    labels = get_labels(all_edges, true_edges)
    model = GATLinkPrediction(embedding_dimension=384, hidden_channels=128, num_layers=2, heads=12)
    decoder = LinkPredictionDecoder()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # print(f'Embedding node 0 iniziale:', embs[0])

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for i, epoch in enumerate(range(num_epochs)):
        model.train()
        data = Data(x=embs, edge_index=true_edges)

        optimizer.zero_grad()  # Resetta i gradienti
        updated_embs = model(data)
        predictions = decoder(updated_embs, all_edges)

        loss = loss_fn(predictions, labels)
        loss.backward()     # Aggiorna i pesi
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
            # print(f'Embedding node 0 a iter {i}:', updated_embs[0])
            # print(f'updated embs in iter {i}: {updated_embs}')
            # print(f'predictions in iter {i}: {predictions}')

    updated_embs = model(data)
    print(f'Embedding node 0:', updated_embs[0])
    print(f'Embedding node 1:', updated_embs[1])
    probabilities = decoder(updated_embs, all_edges)
    predictions = (probabilities >= 0.6).float()

    print_results(probabilities, predictions, labels)
    return model


def main():
    embs = get_embs('../../embeddings/STAC_training_embeddings.json', 0)
    true_edges = get_edges('../../dataset/STAC/train_subindex.json', 0)
    trained_model = train(embs, true_edges, num_epochs=2000, learning_rate=0.001)

    # model = GATLinkPrediction(embedding_dimension=384, hidden_channels=128, num_layers=2, heads=12)
    # test(model, embs, true_edges)

if __name__ == '__main__':
    main()

