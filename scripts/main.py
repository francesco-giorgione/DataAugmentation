import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from scripts.GNN.GAT import *
from scripts.GNN.GraphSAGE import *
from scripts.GNN.GAT import load_models as GAT_load_models
from scripts.GNN.GraphSAGE import load_models as GS_load_models
import copy
import torch
import json
from scripts.LLM.GPT4 import *
from scripts.GNN.models import GATLinkPrediction


def save_all_new_edus(dataset_filename, out_file_path):
    all_dialogues = load_data(dataset_filename)
    all_new_edus = []

    for dialogue_index in range(len(all_dialogues)):
        print(f'Adding dialogue {dialogue_index} to file')

        new_edus, target_node = get_new_edus_gpt4all(all_dialogues, dialogue_index, dataset_name=dataset_filename)
        new_edus_emb = get_new_edus_emb(new_edus)

        curr_dict = dict()
        curr_dict['new_edus'], curr_dict['new_edus_embs'] = [], []
        curr_dict['target_node'] = target_node

        for edu, edu_emb in zip(new_edus, new_edus_emb):
            curr_dict['new_edus'].append(edu)
            curr_dict['new_edus_embs'].append(edu_emb.tolist())

        all_new_edus.append(curr_dict)

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_new_edus, f)
        print(f'New edus saved in file')



def save_all_new_edus_batch(dataset_filename, out_file_path):
    all_dialogues = load_data(dataset_filename)
    all_new_edus = []

    if os.path.exists(out_file_path):
        with open(out_file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    for dialogue_index in range(len(all_dialogues))[307:308]:
        print(f'Adding dialogue {dialogue_index} to file')

        new_edus, target_node = get_new_edus_gpt4all(all_dialogues, dialogue_index, dataset_name=dataset_filename)
        new_edus_emb = get_new_edus_emb(new_edus)

        curr_dict = dict()
        curr_dict['new_edus'], curr_dict['new_edus_embs'] = [], []
        curr_dict['target_node'] = target_node

        for edu, edu_emb in zip(new_edus, new_edus_emb):
            curr_dict['new_edus'].append(edu)
            curr_dict['new_edus_embs'].append(edu_emb.tolist())

        all_new_edus.append(curr_dict)

    existing_data.extend(all_new_edus)

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f)
        print(f'New edus saved in file')


def choose_edus(dataset_filename, embs_filename, edus_file_path, trained_model, trained_link_predictor, out_file_path):
    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    all_new_edus = load_data(edus_file_path)
    new_edus = []

    for dialogue_index in range(len(all_new_edus)):
    # for dialogue_index in range(2):
        target_node = all_new_edus[dialogue_index]['target_node']
        print(f'\nTarget node in dialogue {dialogue_index}: {target_node}')
        print('New edus:', all_new_edus[dialogue_index]['new_edus'])

        curr_dialogue_values = []

        for i_emb in range(len(all_new_edus[dialogue_index]['new_edus'])):
            curr_new_edu_value = trained_model.predict(
                dialogue_json=all_dialogues[dialogue_index],
                old_embs=torch.tensor([item['embedding'] for item in all_embs[dialogue_index]], dtype=torch.float),
                target_node=target_node,
                new_edus_emb=all_new_edus[dialogue_index]['new_edus_embs'][i_emb],
                model=trained_model,
                link_predictor=trained_link_predictor
            )

            curr_dialogue_values.append(curr_new_edu_value)

        dialogue_probs = curr_dialogue_values
        best_new_edu_index = get_best_new_edu_index(dialogue_probs)
        best_new_edu = all_new_edus[dialogue_index]['new_edus'][best_new_edu_index]
        new_edus.append({
            'target_node': target_node,
            'new_edu': best_new_edu
        })

    print('new_edus:', new_edus)

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(new_edus, f)
        print(f'Best new edus saved successfully')


def augment(dataset_filename, new_edus_file_path, out_file_path):
    all_dialogues = load_data(dataset_filename)
    n = len(all_dialogues)
    new_edus = load_data(new_edus_file_path)

    for dialogue_index, dialogue_info in enumerate(new_edus):
        target_node = dialogue_info['target_node']
        new_edu = dialogue_info['new_edu']
        new_dialogue = copy.deepcopy(all_dialogues[dialogue_index])
        new_dialogue['edus'][target_node]['text'] = new_edu
        new_dialogue['id'] = f"{n + dialogue_index}"
        all_dialogues.append(new_dialogue)

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_dialogues, f)
        print(f'Augmented dataset created successfully')


class DataAugmentationPipeline:
    def __init__(self, dataset_name, dataset_filename, trained_GNN, trained_link_predictor, all_new_edus_available=True):
        self.model = None
        self.dataset_name = dataset_name
        self.dataset_filename = dataset_filename
        self.trained_GNN = trained_GNN
        self.trained_link_predictor = trained_link_predictor
        self.all_new_edus_available = all_new_edus_available


    def __call__(self):
        if not self.all_new_edus_available:
            save_all_new_edus(
                dataset_filename=self.dataset_filename,
                out_file_path=f'../new_edus/{self.dataset_name}_new_edus.json'
            )

        choose_edus(
            dataset_filename=self.dataset_filename,
            embs_filename=f'../embeddings/MPNet/{self.dataset_name}_training_embeddings.json',
            edus_file_path=f'../new_edus/{self.dataset_name}_new_edus.json',
            trained_model=self.trained_GNN,
            trained_link_predictor=trained_link_predictor,
            out_file_path=f'../new_edus/best/{self.dataset_name}_best_edus.json'
        )

        augment(
            dataset_filename=self.dataset_filename,
            new_edus_file_path=f'../new_edus/best/{self.dataset_name}_best_edus.json',
            out_file_path=f'../augmented_datasets/{self.dataset_name}_augmented.json'
        )


if __name__ == '__main__':
    STAC_dataset_filename = '../dataset/STAC/train_subindex.json'
    MOLWENI_dataset_filename = '../dataset/MOLWENI/train.json'
    MINECRAFT_dataset_filename = '../dataset/MINECRAFT/TRAIN_307_bert.json'
    STAC_embs_filename = '../embeddings/MPNet/STAC_training_embeddings.json'
    MOLWENI_embs_filename = '../embeddings/MPNet/MOLWENI_training_embeddings.json'
    MINECRAFT_embs_filename = '../embeddings/MPNet/MINECRAFT_training_embeddings.json'
    STAC_edus_file_path = '../new_edus/STAC_new_edus.json'
    MOLWENI_edus_file_path = '../new_edus/MOLWENI_new_edus.json'
    MINECRAFT_edus_file_path = '../new_edus/MINECRAFT_new_edus.json'
    STAC_best_edus_file_path = '../new_edus/best/STAC_best_edus.json'
    MOLWENI_best_edus_file_path = '../new_edus/best/MOLWENI_best_edus.json'
    MINECRAFT_best_edus_file_path = '../new_edus/best/MINECRAFT_best_edus.json'
    STAC_augmented_path = '../augmented_datasets/STAC_augmented.json'
    MOLWENI_augmented_path = '../augmented_datasets/MOLWENI_augmented.json'
    MINECRAFT_augmented_path = '../augmented_datasets/MINECRAFT_augmented.json'

    trained_model, trained_link_predictor = GS_load_models('../pretrain_model_GS/Minecraft_pretrained_models_1.pth', num_layers=3)
    # trained_model, trained_link_predictor = GAT_load_models('../pretrain_model_GAT/pretrained_models_MINECRAFT.pth')

    dataAugmenter = DataAugmentationPipeline(
        dataset_name='MINECRAFT',
        dataset_filename=MINECRAFT_dataset_filename,
        trained_GNN=trained_model,
        trained_link_predictor=trained_link_predictor,
        all_new_edus_available=True
    )

    dataAugmenter()

