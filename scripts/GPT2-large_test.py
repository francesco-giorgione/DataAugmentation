import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from graph_builder import *
import networkx as nx
from networkx.readwrite import json_graph


model_name = "gpt2-large"  # Puoi scegliere modelli più piccoli come 'gpt2-medium' se vuoi
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def get_response(prompt):
    # Tokenizza il prompt
    input_ids = tokenizer.encode(
        prompt,
        return_tensors='pt'
    )

    # Maschera di attention che fa in modo che solo i token reali vengano considerati (token di padding scartati)
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
    pad_token_id = model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else None

    # Genera una continuazione del testo
    output = model.generate(
        input_ids,
        max_length=input_ids.size(1) + 30,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.2,
        temperature=0.5,
        do_sample=True,
        attention_mask=attention_mask,
        pad_token_id = model.config.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


if __name__ == '__main__':
    instruction = "Please paraphrase the following sentence with more creativity."
    text = "\nI feel bad today"
    command = "\nParaphrase:"

    # prompt = instruction + text + command

    prompt = """Please paraphrase the following sentence in a different and concise way, keeping the meaning the same.
    Original sentence: 'One of the challenges in software system development is to understand information security requirements that need to be fulfilled.'
    Paraphrase: """

    prompt_2 = """Please paraphrase the following sentence in a more creative way.
    Original sentence: 'It's cloudy today'
    Paraphrase: """

    # Carica i dati dal file JSON
    input_json = carica_json_da_file('dataset/STAC/train_subindex.json')

    # Creiamo e visualizziamo un grafo per ogni dialogo nel JSON
    
    
    graph = crea_grafo_da_json([input_json[0]])  # Passiamo il singolo dialogo come lista
    # visualizza_grafo_dag(graph, 1)  # Visualizziamo il grafo del dialogo
    target_node = 2
    if target_node in graph:
        # Ottieni i discendenti (nodi raggiungibili dal target)
        reachable_nodes = nx.descendants(graph, target_node)
        reachable_nodes.add(target_node)  # Includi il nodo target stesso

        # Ottieni gli antenati (nodi che possono raggiungere il target)
        ancestor_nodes = nx.ancestors(graph, target_node)
        ancestor_nodes.add(target_node)  # Includi il nodo target stesso

        # Combina i nodi raggiungibili e gli antenati
        all_relevant_nodes = reachable_nodes.union(ancestor_nodes)

        # Estrai il sottografo
        subgraph = graph.subgraph(ancestor_nodes)
        
        # Visualizza il sottografo
        visualizza_grafo_dag(subgraph, 1)

        # Stampa i nodi e gli archi del grafo
        print("Nodi nel grafo:", graph.nodes())
        print("Archi nel grafo:", graph.edges())

        # Stampa i nodi e gli archi del sottografo
        print("Nodi nel sottografo:", subgraph.nodes())
        print("Archi nel sottografo:", subgraph.edges())
    else:
        print(f"Il nodo {target_node} non è presente nel grafo.")

    utt_subgraph = ""
    for edu, att in subgraph.nodes(data=True):
        utt_subgraph += f"Utterance {edu} - {att["speaker"]}: {att["text"]}\n"

    relations = ""
    for nodo1, nodo2, att in subgraph.edges(data=True):
        relations += f"({nodo1}, {nodo2}) - {att['relationship']}\n"

    utt_subgraph += "Relations: " + relations

    print(utt_subgraph)
    prompt_graph = f"Generates a new utterance in place of the {target_node} utterance, without changing the meaning of the utterance and without changing the relations between the utterances. Dialogue: \n{utt_subgraph} \n Utterance {target_node}: "

    response = get_response(prompt_graph)

    print(f'\n\n{response}')