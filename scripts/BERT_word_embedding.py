import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from vincoli_edu import *

def create_word_embeddings(edus_list):
    # Set a random seed
    random_seed = 42
    random.seed(random_seed)

    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    # Tokenize and encode the sentences in batch
    inputs = tokenizer(
        edus_list, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract word embeddings
    word_embeddings = outputs.last_hidden_state 

    print(f"Shape of word embeddings: {word_embeddings.shape}")
    return word_embeddings

def main():
    file_path = "dataset/STAC/train_subindex.json"
    data = load_data(file_path)

    edus_list = get_edus_list(data)

    print(f"Number of EDUs: {len(edus_list)}")
    print(edus_list[:5])

    embeddings = create_word_embeddings(edus_list)
    #print(embeddings)

if __name__ == "__main__":
    main()
