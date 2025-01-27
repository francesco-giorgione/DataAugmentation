from sentence_transformers import SentenceTransformer, SimilarityFunction
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.constraints.vincoli_edu import *


def create_embeddings(dialogue_edus_list):

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Lista per memorizzare le coppie (contatore, lista di embeddings)
    embeddings_edu_list = []
    i = 0
    # Itera su ogni dialogo (contatore, lista di edus)
    for edus_list in dialogue_edus_list:
        # Calcola gli embeddings per la lista di EDUs
        embeddings = model.encode(edus_list, 
                                  show_progress_bar=True)

        print(f"Embedding per il dialogo {i}: {embeddings.shape}")

        # Crea la lista di coppie (edu, embedding) per questo dialogo
        embeddings_edu_list.append([(edu, embedding) for edu, embedding in zip(edus_list, embeddings)])
        i +=1
    
    return embeddings_edu_list


def save_embeddings(embeddings_edu_list, filename):
    
    # Crea la directory se non esiste
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Creazione della lista di dizionari con contatore e lista di EDU-embedding
    data_json = []
    
    for edus_embeddings in embeddings_edu_list:
        # Per ogni dialogo, aggiungi un dizionario con il contatore e la lista di EDU-embedding
        data_json.append(
                [{"edu": edu, "embedding": embedding.tolist()} 
                for edu, embedding in edus_embeddings]
            )

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_json, f)
        

def main():

    dataset_name_list = ["STAC_training", "MOLWENI_training", "MINECRAFT_training", 
                         "STAC_testing", "MOLWENI_testing", "MINECRAFT_testing101", "MINECRAFT_testing133",
                         "MOLWENI_val", "MINECRAFT_val32", "MINECRAFT_val100"]

    #dataset_name_list = ["MOLWENI_training", "MOLWENI_testing"]
            

    for dataset_name in dataset_name_list:
        if dataset_name == "STAC_training":
            file_path = "dataset/STAC/train_subindex.json"
        elif dataset_name == "MOLWENI_training":
            file_path = "dataset/MOLWENI/train.json"
        elif dataset_name == "MINECRAFT_training":
            file_path = "dataset/MINECRAFT/TRAIN_307_bert.json"

        elif dataset_name == "STAC_testing":
            file_path = "dataset/STAC/test_subindex.json"
        elif dataset_name == "MOLWENI_testing":
            file_path = "dataset/MOLWENI/test.json"
        elif dataset_name == "MINECRAFT_testing101":
            file_path = "dataset/MINECRAFT/TEST_101_bert.json"
        elif dataset_name == "MINECRAFT_testing133":
            file_path = "dataset/MINECRAFT/TEST_133.json"

        elif dataset_name == "MOLWENI_val":
            file_path = "dataset/MOLWENI/dev.json"
        elif dataset_name == "MINECRAFT_val32":
            file_path = "dataset/MINECRAFT/DEV_32_bert.json"
        elif dataset_name == "MINECRAFT_val100":
            file_path = "dataset/MINECRAFT/VAL_100_bert.json"


        data = load_data(file_path)

        edus_list = get_dialogue_edus_list(data)

        print(f"Numero di dialoghi del dataset " + dataset_name + f": : {len(edus_list)}")
        #print(edus_list[:5])

        embeddings_edu_list = create_embeddings(edus_list)
        # print(embeddings_edu_list[:5])
        # print(embeddings.shape)

        # esempio = [":)"]
        # embedding_esempio = create_embeddings(esempio)
        # similarities = model.similarity(embeddings_edu_list[0][1], embedding_esempio[0][1])
        # print(similarities)

        save_embeddings(embeddings_edu_list, f"embeddings/MiniLM/{dataset_name}_embeddings.json")
    

if __name__ == "__main__":
    main()




