import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from GAT import GATLinkPrediction, LinkPredictionDecoder, LinkPredictionDecoderKernel, LinkPredictionMLP
import random


def test_json(embs_filename, dialogue_index):
    with open(embs_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)[dialogue_index]

    for item in data:
        print(item)
        print('\n\n\n\n')


def n_dialogues(dataset_filename):
    with open(dataset_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return len(data)


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


def loss_fn(pred, labels, pos_weight=2.0, neg_weight=1.0):
    weights = torch.where(labels == 1, pos_weight, neg_weight)
    loss = F.binary_cross_entropy(pred, labels, weight=weights)
    return loss


def get_metrics(predictions, labels):
    TP = torch.sum((predictions == 1) & (labels == 1)).item()
    FP = torch.sum((predictions == 1) & (labels == 0)).item()
    TN = torch.sum((predictions == 0) & (labels == 0)).item()
    FN = torch.sum((predictions == 0) & (labels == 1)).item()

    accuracy = (TP + TN) / len(predictions)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return accuracy, precision, recall


def results(probabilities, predictions, labels):
    # print('Probabilities:', probabilities)
    # print('Predictions:', predictions)
    # print('Labels:', labels)

    loss = loss_fn(probabilities, labels)
    accuracy, precision, recall = get_metrics(predictions, labels)
    return loss, accuracy, precision, recall


def test(model, embs, true_edges):
    all_edges = get_all_edges(embs)
    labels = get_labels(all_edges, true_edges)

    model.eval()
    data = Data(x=embs, edge_index=true_edges)

    with torch.no_grad():
        updated_embs = model(data)

    # print(f'Embedding node 0:', updated_embs[0])
    # print(f'Embedding node 1:', updated_embs[1])

    decoder = LinkPredictionDecoder()
    probabilities = decoder(updated_embs, all_edges)
    predictions = (probabilities >= 0.7).float()

    loss, accuracy, precision, recall = results(probabilities, predictions, labels)
    print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')


def old_train(model, embs, true_edges, num_epochs=100, learning_rate=0.01):
    all_edges = get_all_edges(embs)
    labels = get_labels(all_edges, true_edges)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
            # print(f'Embedding node 0 a iter {i}:', updated_embs[0])
            # print(f'updated embs in iter {i}: {updated_embs}')
            # print(f'predictions in iter {i}: {predictions}')

    updated_embs = model(data)
    print(f'Embedding node 0:', updated_embs[0])
    print(f'Embedding node 1:', updated_embs[1])
    probabilities = decoder(updated_embs, all_edges)
    predictions = (probabilities >= 0.7).float()

    loss, accuracy, precision, recall = results(probabilities, predictions, labels)
    print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')
    return model


def one_worker(dataset_filename, embs_filename):
    embs = get_embs(embs_filename, 12)
    true_edges = get_edges(dataset_filename, 12)
    # model = GATLinkPrediction(embedding_dimension=384, hidden_channels=128, num_layers=2, heads=12)
    # trained_model = old_train(model, embs, true_edges, num_epochs=100, learning_rate=0.001)

    model = GATLinkPrediction(embedding_dimension=384, hidden_channels=128, num_layers=2, heads=12)
    test(model, embs, true_edges)


def test_batch(model, dataset_filename, embs_filename, batch_size=10, decoder=None):
    if decoder is None:
        decoder = LinkPredictionDecoder()

    n = n_dialogues(dataset_filename)
    num_samples = min(batch_size, n)
    indices = random.sample(range(n), num_samples)

    total_loss, total_accuracy, total_precision, total_recall = 0, 0, 0, 0

    for i in indices:
        embs = get_embs(embs_filename, i)
        true_edges = get_edges(dataset_filename, i)
        all_edges = get_all_edges(embs)
        labels = get_labels(all_edges, true_edges)

        model.eval()
        data = Data(x=embs, edge_index=true_edges)

        with torch.no_grad():
            updated_embs = model(data)

        probabilities = decoder(updated_embs, all_edges)
        predictions = (probabilities >= 0.7).float()

        print(f'Dialogue {i} included for validation')
        curr_loss, curr_accuracy, curr_precision, curr_recall = results(probabilities, predictions, labels)
        total_loss += curr_loss
        total_accuracy += curr_accuracy
        total_precision += curr_precision
        total_recall += curr_recall

    loss = total_loss/batch_size
    accuracy = total_accuracy/batch_size
    precision = total_precision/batch_size
    recall = total_recall/batch_size

    print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')


def train_batch(model, dataset_filename, embs_filename, num_epochs=100, learning_rate=0.001, batch_size=10, decoder=None):
    if decoder is None:
        decoder = LinkPredictionDecoder()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        sum_loss = 0
        n = n_dialogues(dataset_filename)

        num_samples = min(batch_size, n)
        indices = random.sample(range(n), num_samples)
        print(f'n dialogues: {n}')
        print(f'indices: {indices}')

        for j in indices:
            print(f'Epoch {epoch + 1}, Dialogue {j}')
            embs = get_embs(embs_filename, j)
            true_edges = get_edges(dataset_filename, j)
            all_edges = get_all_edges(embs)
            labels = get_labels(all_edges, true_edges)

            model.train()
            data = Data(x=embs, edge_index=true_edges)

            optimizer.zero_grad()  # Resetta i gradienti
            updated_embs = model(data)
            probabilities = decoder(updated_embs, all_edges)

            sum_loss += loss_fn(probabilities, labels)

        loss = sum_loss / num_samples   # Media della loss sui campioni dell'i-esima epoca
        loss.backward()     # Aggiorna i pesi
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            # predictions = (probabilities >= 0.7).float()
            # print_results(probabilities, predictions, labels)

            break

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")

    return model


def all_worker(training_dataset_filename, training_embs_filename, testing_dataset_filename=None, testing_embs_filename=None):
    model = GATLinkPrediction(embedding_dimension=768, hidden_channels=128, num_layers=2, heads=12)
    trained_model = train_batch(model, training_dataset_filename, training_embs_filename, batch_size=50, decoder=LinkPredictionDecoderKernel(0.7))
    test_batch(trained_model, testing_dataset_filename, testing_embs_filename, batch_size=50, decoder=LinkPredictionDecoderKernel(0.7))


def main():
    # one_worker('../../dataset/STAC/train_subindex.json', '../../embeddings/STAC_training_embeddings.json')
    # all_worker('../../dataset/STAC/train_subindex.json', '../../embeddings/MPNet/STAC_training_embeddings.json',
    #            '../../dataset/STAC/test_subindex.json', '../../embeddings/MPNet/STAC_testing_embeddings.json')
    all_worker('../../dataset/MOLWENI/test.json', '../../embeddings/MPNet/MOLWENI_testing_embeddings.json',
              '../../dataset/MOLWENI/test.json', '../../embeddings/MPNet/MOLWENI_testing_embeddings.json')

if __name__ == '__main__':
    main()

