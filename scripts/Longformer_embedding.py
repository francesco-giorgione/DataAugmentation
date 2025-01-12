from transformers import LongformerTokenizer, LongformerModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from vincoli_edu import *

def create_word_embeddings(edus_dialogue_list, batch_size):
    # Carica Longformer tokenizer e modello
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)

    all_embeddings = []  # Lista per memorizzare gli embeddings di tutti i batch

    # Suddividi il dataset in batch più piccoli
    num_batches = len(edus_dialogue_list) // batch_size + (1 if len(edus_dialogue_list) % batch_size != 0 else 0)

    for i in range(num_batches):
        # Estrai il batch corrente
        batch = edus_dialogue_list[i * batch_size : (i + 1) * batch_size]

        # Tokenizza le frasi
        inputs = tokenizer(batch, 
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True,
                           max_length=4096)

        # Calcola gli embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Estrai gli embedding delle parole
        word_embeddings = outputs.last_hidden_state

        print(word_embeddings.shape)

        # Aggiungi gli embeddings di questo batch alla lista globale
        all_embeddings.append(word_embeddings)

        print(f"Processed batch {i + 1}/{num_batches}, batch size: {len(batch)}")

    # Concatenate gli embeddings di tutti i batch in un'unica tensor
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    return all_embeddings_tensor


def main():
    file_path = "dataset/MOLWENI/train.json"
    data = load_data(file_path)

    edus_dialogue_list = get_edus_dialogue_list(data)

    print(f"Numero di dialoghi: {len(edus_dialogue_list)}")
    #print(edus_dialogue_list[:5])

    # len_edus_dialogue = []

    # for edu in edus_dialogue_list:
    #     len_edus_dialogue.append(len(edu))

    # # Trova il massimo e la posizione
    # max_len = max(len_edus_dialogue)
    # max_index = len_edus_dialogue.index(max_len)# Trova il massimo e la posizione
    # max_len = max(len_edus_dialogue)
    # max_index = len_edus_dialogue.index(max_len)

    # print(f"Valore massimo: {max_len}")
    # print(f"Posizione nella lista: {max_index}")

    # print(f"Valore massimo: {max_len}")
    # print(f"Posizione nella lista: {max_index}")

    batch_size=256

    embeddings = create_word_embeddings(edus_dialogue_list[:512], batch_size)
    print(embeddings.shape)  # Stampa la forma finale degli embeddings
    print(embeddings[0].size())  # Forma del primo esempio
    print(embeddings[1].size())  # Forma del secondo esempio
    
    # Salvataggio in un file
    torch.save(embeddings, "output/Longformer_tensor_embeddings_molweni.pt")

    # # Caricamento dal file
    # loaded_tensor = torch.load("output/tensor_Longformer.pt")

    # decoded_text = tokenizer.decode(loaded_tensor[0, 0, :], skip_special_tokens=True)
    
    # print(decoded_text)

if __name__ == "__main__":
    main()
