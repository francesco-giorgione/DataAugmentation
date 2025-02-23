import torch
from statistics import mean
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import random
from scripts.GNN.utils import *
import sys
import os
from scripts.GNN.models import GATLinkPrediction, LinkPredictorMLP
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def load_models(path):
    model = GATLinkPrediction(embedding_dimension=768, hidden_channels=384, num_layers=3, dropout=0.5, heads=32)
    link_predictor = LinkPredictorMLP(in_channels=384, hidden_channels=192, out_channels=1, num_layers=3, dropout=0.5)


    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)  # Usa 'cuda' se hai una GPU
    model.load_state_dict(checkpoint['model_state_dict'])
    link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])

    print(f"Modelli caricati da {path}")
    return model, link_predictor



def validate_worker(model, predictor, emb, edge_index, pos_test_edge):
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

    loss = -torch.log(pos_test_pred + 1e-15).mean() - torch.log(1 - neg_test_pred + 1e-15).mean()

    # print(f'pos test pred {pos_test_pred}\nneg test pred {neg_test_pred}')
    return loss.item(), pos_test_pred, neg_test_pred


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


def old_validate(dataset_filename, embs_filename, model, link_predictor, batch_size=1000):
    total_accuracy, total_precision, total_recall = 0, 0, 0
    batch_losses = []

    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    n = len(all_dialogues)
    num_samples = min(batch_size, n)
    sampled_dialogues = random.sample(range(n), num_samples)

    for dialogue_index in sampled_dialogues:
        embs = torch.tensor([item['embedding'] for item in all_embs[dialogue_index]], dtype=torch.float)
        edge_index = super_new_get_edges(all_dialogues, dialogue_index)

        print(f'[Testing] Sampled dialogue {dialogue_index}')
        accuracy, precision, recall = validate_worker(model, link_predictor, embs, edge_index, edge_index)

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall

    accuracy = total_accuracy / num_samples
    precision = total_precision / num_samples
    recall = total_recall / num_samples
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')


def validate(dataset_filename, embs_filename, model, link_predictor, batch_size=50, threshold = 0.5, loss_path="", loss_desc="", isTest = False):
    total_accuracy, total_precision, total_recall = 0, 0, 0
    pos_preds, neg_preds = [], []
    batch_losses = []

    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    n = len(all_dialogues)
    num_samples = min(batch_size, n)
    sampled_dialogues = random.sample(range(n), num_samples)
    test_loder = DataLoader([i for i in range(n)], batch_size=batch_size, shuffle=True)
    with tqdm(total=len(test_loder), desc="Validation") as progress_bar:
        for i_batch, sampled_dialogues in enumerate(test_loder, start=1):
            dialogue_losses = []

            for dialogue_index in sampled_dialogues:
                embs = torch.tensor([item['embedding'] for item in all_embs[dialogue_index]], dtype=torch.float)
                edge_index = super_new_get_edges(all_dialogues, dialogue_index)

                # print(f'[Testing] Sampled dialogue {dialogue_index}')
                curr_loss, curr_pos_pred, curr_neg_pred = validate_worker(model, link_predictor, embs, edge_index, edge_index)
                # print(f'[Batch {i_batch}] Loss: {curr_loss}')

                pos_preds.append(curr_pos_pred)
                neg_preds.append(curr_neg_pred)
                dialogue_losses.append(curr_loss)

            progress_bar.update(1)
            if not isTest:
                loss = mean(dialogue_losses)
                batch_losses.append(loss)
                progress_bar.set_postfix({'Loss': loss})

    pos_preds = torch.cat(pos_preds, dim=0)
    neg_preds = torch.cat(neg_preds, dim=0)
    accuracy, precision, recall = eval_metrics(pos_preds, neg_preds, threshold = threshold)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
    if not isTest:
        plot_loss(batch_losses, len(batch_losses), loss_path, loss_desc)



def link_validate(dataset_filename, embs_filename, model, link_predictor, threshold=0.5):
    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    n = len(all_dialogues)

    total_predictions, total_correct_predictions = 0, 0

    for dialogue_index in range(n):
        n_edus = len(all_dialogues[dialogue_index]['edus'])
        curr_correct_prediction_counter = 0
        embs = torch.tensor([item['embedding'] for item in all_embs[dialogue_index]], dtype=torch.float)

        edge_index = super_new_get_edges(all_dialogues, dialogue_index)
        target_node = get_target_node(dataset_filename, create_graph(all_dialogues[dialogue_index]))[0]
        removed_edge_index, filtered_edge_index = filter_edge_index(edge_index, target_node)

        # print('Target node', target_node)
        # print('Edge index', edge_index)
        # print('Removed index', removed_edge_index)
        # print('Filtered index', filtered_edge_index)

        # data = Data(x=embs, edge_index=edge_index)   # (N, d)
        data = Data(x=embs, edge_index=filtered_edge_index)
        node_embs = model(data)

        for i in range(removed_edge_index.shape[1]):
            emb_1 = node_embs[removed_edge_index[0, i]]
            emb_2 = node_embs[removed_edge_index[1, i]]
            # print(emb_1, emb_2)

            prob = link_predictor(emb_1, emb_2)
            # print(prob)

            if prob >= threshold:
                curr_correct_prediction_counter += 1

        print(f'[Dialogo {dialogue_index}] Predizioni corrette: {curr_correct_prediction_counter}/{removed_edge_index.shape[1]}')
        total_predictions += removed_edge_index.shape[1]
        total_correct_predictions += curr_correct_prediction_counter

    print(f'Totale predizioni corrette: {total_correct_predictions}/{total_predictions} '
          f'({(total_correct_predictions / total_predictions) * 100:.2f}%)')



def train(dataset_filename, embs_filename, loss_path, loss_desc, num_epochs=100, batch_size=50, learning_rate=0.001, model=None, link_predictor=None):
    if model is None:
        model = GATLinkPrediction(embedding_dimension=768, hidden_channels=256, num_layers=2, heads=16)

    if link_predictor is None:
        link_predictor = LinkPredictorMLP(in_channels=256, hidden_channels=256, out_channels=1, num_layers=4, dropout=0.5)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=learning_rate)

    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    n = len(all_dialogues)
    loss_history = []

    with tqdm(total=num_epochs, desc="Training") as progress_bar:
        for epoch in range(num_epochs):
            batch_losses = []

            for batch_index, batch_dialogues in enumerate(DataLoader(range(n), batch_size=batch_size, shuffle=True, num_workers=4), start=1):
                dialogue_losses = []
                batch_dialogues = batch_dialogues.tolist()
                # print('Sampled dialogues:', batch_dialogues)

                for dialogue_index in batch_dialogues:
                    embs = torch.tensor([item['embedding'] for item in all_embs[dialogue_index]], dtype=torch.float)
                    edge_index = super_new_get_edges(all_dialogues, dialogue_index)

                    dialogue_losses.append(train_worker(model, link_predictor, embs, edge_index, edge_index, batch_size, optimizer))

                batch_loss = torch.stack(dialogue_losses).mean()
                batch_losses.append(batch_loss.item())

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_index} -> Batch training Loss: {batch_loss:.4f}")

            # print(f'Num batch in epoch {epoch}: {batch_index}')
            # epoch_loss = torch.stack(batch_losses).mean()
            epoch_loss = mean(batch_losses)
            loss_history.append(epoch_loss)
            # print(f"Epoch {epoch+1}/{num_epochs}, Training epoch loss: {epoch_loss:.4f}")

            save_models(model, link_predictor, file_path)
            progress_bar.update(1)
            progress_bar.set_postfix({'Loss': epoch_loss})

    plot_loss(loss_history, num_epochs, loss_path, loss_desc, isTrain=True)
    return model, link_predictor




if __name__ == '__main__':

    trained_model = GATLinkPrediction(embedding_dimension=768, hidden_channels=384, num_layers=2, dropout=0.5, heads=32)
    trained_link_predictor = LinkPredictorMLP(in_channels=384, hidden_channels=192, out_channels=1, num_layers=2, dropout=0.5)
    # file_path = '../../pretrain_model_GAT/pretrained_models_STAC.pth'
    # file_path = '../../pretrain_model_GAT/pretrained_models_MINECRAFT.pth'
    file_path = '../../pretrain_model_GAT/pretrained_models_MOLWENI.pth'

    trained_model, trained_link_predictor = load_models(file_path)

    # STAC    
    """ trained_model, trained_link_predictor = train('../../dataset/STAC/train_subindex.json',
                        "../../embeddings/MPNet/STAC_training_embeddings.json", 
                        "../../plot_loss/GAT_STAC_train5.png", "STAC Training Loss", 
                        num_epochs=30, batch_size=32, learning_rate=0.001, model=trained_model, link_predictor=trained_link_predictor)
    """
    # --- VALIDAZIONE ---
    """ validate('../../dataset/STAC/dev.json', '../../embeddings/MPNet/STAC_val_embeddings.json', 
            trained_model, trained_link_predictor, batch_size=32, 
            loss_path = "../../plot_loss/GAT_STAC_val.png", loss_desc = "STAC Validation Loss") """

    # --- TESTING ---
    """ validate('../../dataset/STAC/test_subindex.json', '../../embeddings/MPNet/STAC_testing_embeddings.json', 
            trained_model, trained_link_predictor, batch_size=32, isTest=True) """



    # MINECRAFT
    """ 
    trained_model, trained_link_predictor = train('../../dataset/MINECRAFT/TRAIN_307_bert.json',
                        "../../embeddings/MPNet/MINECRAFT_training_embeddings.json", 
                        "../../plot_loss/GAT_MINECRAFT_train4.png", "MINECRAFT Training Loss", 
                        num_epochs=30, batch_size=32, learning_rate=0.001, model=trained_model, link_predictor=trained_link_predictor)
    """
    # --- VALIDAZIONE ---
    """ validate('../../dataset/MINECRAFT/VAL_all.json', '../../embeddings/MPNet/MINECRAFT_val_embeddings.json', 
            trained_model, trained_link_predictor, batch_size=32, threshold = 0.5,
            loss_path = "../../plot_loss/GAT_MINECRAFT_val.png", loss_desc = "MINECRAFT Validation Loss") """
    
    # --- TESTING ---
    """ validate('../../dataset/MINECRAFT/TEST_133.json', '../../embeddings/MPNet/MINECRAFT_testing133_embeddings.json', 
            trained_model, trained_link_predictor, batch_size=32, isTest=True) """

    # MOLWENI    
    """ 
    trained_model, trained_link_predictor = train('../../dataset/MOLWENI/train.json',
                        "../../embeddings/MPNet/MOLWENI_training_embeddings.json", 
                        "../../plot_loss/GAT_MOLWENI_train.png", "MOLWENI Training Loss", 
                        num_epochs=10, batch_size=32, learning_rate=0.001, model=trained_model, link_predictor=trained_link_predictor)
    """
    # --- VALIDAZIONE ---
    """ validate('../../dataset/MOLWENI/dev.json', '../../embeddings/MPNet/MOLWENI_val_embeddings.json',
            trained_model, trained_link_predictor, batch_size = 32,
            loss_path = "../../plot_loss/GAT_MOLWENI_val.png", loss_desc = "MOLWENI Validation Loss") """
    
    # --- TESTING ---
    """ validate('../../dataset/MOLWENI/test.json', '../../embeddings/MPNet/MOLWENI_testing_embeddings.json', 
            trained_model, trained_link_predictor, batch_size=32, isTest=True) """
    

    # link_validate('../../dataset/MOLWENI/dev.json', '../../embeddings/MPNet/MOLWENI_val_embeddings.json', trained_model, trained_link_predictor)

    # predict('../../dataset/MINECRAFT/VAL_all.json', '../../embeddings/MPNet/MINECRAFT_val_embeddings.json', trained_model, trained_link_predictor)
    # predict('../../dataset/MOLWENI/dev.json', '../../embeddings/MPNet/MOLWENI_val_embeddings.json', trained_model, trained_link_predictor)



