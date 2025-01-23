from GNN_new_test import *
from utils import *
import openai
import sys
import os
import torch
import json
import time
from transformers import Pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GPT4_test import get_new_edus_emb, get_new_edus, get_new_edus_gpt4all





def save_all_new_edus(dataset_filename, out_file_path):
    all_dialogues = load_data(dataset_filename)
    all_new_edus = []

    for dialogue_index in range(len(all_dialogues)):
        print(f'Adding dialogue {dialogue_index} to file')
        wait_time = 10

        while True:
            try:
                new_edus, target_node = get_new_edus(all_dialogues, dialogue_index, dataset_name=dataset_filename)
                break
            except Exception as e:
                wait_time += 20
                print(f"Rate limit superato! Aspetto {wait_time}s prima di riprovare...")
                time.sleep(wait_time)

                os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
                with open(out_file_path, 'w', encoding='utf-8') as f:
                    json.dump(all_new_edus, f)
                    print(f'New edus saved in file until dialogue {dialogue_index - 1}')

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


def choose_edus(dataset_filename, embs_filename, edus_file_path, trained_model, out_file_path):
    all_dialogues = load_data(dataset_filename)
    all_embs = load_data(embs_filename)
    all_new_edus = load_data(edus_file_path)
    new_edus = []

    # for dialogue_index in range(len(all_new_edus)):
    for dialogue_index in range(2):
        target_node = all_new_edus[dialogue_index]['target_node']
        print(f'\nTarget node in dialogue {dialogue_index}: {target_node}')
        print('New edus:', all_new_edus[dialogue_index]['new_edus'])

        curr_dialogue_values = []

        for i_emb in range(len(all_new_edus[dialogue_index]['new_edus'])):
            curr_new_edu_value = predict(
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
        new_dialogue = all_dialogues[dialogue_index]
        new_dialogue['edus'][target_node]['text'] = new_edu
        new_dialogue['id'] = n + dialogue_index
        all_dialogues.append(new_dialogue)

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_dialogues, f)
        print(f'Augmented dataset created successfully')


class MyPipeline(Pipeline):
    function_list = [save_all_new_edus, save_all_new_edus_batch, choose_edus, augment]

    def forward(self, model_name):
        pass



if __name__ == '__main__':
    STAC_dataset_filename = '../../dataset/STAC/train_subindex.json'
    MOLWENI_dataset_filename = '../../dataset/MOLWENI/train.json'
    MINECRAFT_dataset_filename = '../../dataset/MINECRAFT/TRAIN_307_bert.json'
    STAC_embs_filename = '../../embeddings/MPNet/STAC_training_embeddings.json'
    MOLWENI_embs_filename = '../../embeddings/MPNet/MOLWENI_training_embeddings.json'
    MINECRAFT_embs_filename = '../../embeddings/MPNet/MINECRAFT_training_embeddings.json'
    STAC_edus_file_path = '../../new_edus/STAC_new_edus.json'
    MOLWENI_edus_file_path = '../../new_edus/MOLWENI_new_edus.json'
    MINECRAFT_edus_file_path = '../../new_edus/MINECRAFT_new_edus.json'
    STAC_best_edus_file_path = '../../new_edus/best/STAC_best_edus.json'
    MOLWENI_best_edus_file_path = '../../new_edus/best/MOLWENI_best_edus.json'
    MINECRAFT_best_edus_file_path = '../../new_edus/best/MINECRAFT_best_edus.json'
    STAC_trained_models_file_path = 'pretrained_models_STAC.pth'
    MOLWENI_trained_models_file_path = 'pretrained_models_MOLWENI.pth'
    MINECRAFT_trained_models_file_path = 'pretrained_models_MINECRAFT.pth'
    STAC_augmented_path = '../../augmented_datasets/STAC_augmented.json'
    MOLWENI_augmented_path = '../../augmented_datasets/MOLWENI_augmented.json'
    MINECRAFT_augmented_path = '../../augmented_datasets/MINECRAFT_augmented.json'
    trained_model, trained_link_predictor = load_models(MOLWENI_trained_models_file_path)

    # save_all_new_edus_batch(
    #     dataset_filename=MINECRAFT_dataset_filename,
    #     out_file_path=MINECRAFT_edus_file_path
    # )
    #
    # choose_edus(
    #     dataset_filename=MOLWENI_dataset_filename,
    #     embs_filename=MOLWENI_embs_filename,
    #     edus_file_path=MOLWENI_edus_file_path,
    #     trained_model=trained_model,
    #     out_file_path=MOLWENI_best_edus_file_path
    # )

    augment(
        dataset_filename=MOLWENI_dataset_filename,
        new_edus_file_path=MOLWENI_best_edus_file_path,
        out_file_path=MOLWENI_augmented_path
    )

