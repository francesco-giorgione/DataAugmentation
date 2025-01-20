from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import re
from get_edu import *
from vincoli_edu import *
import random

model_name = "meta-llama/Llama-2-7b-chat-hf"

HUGGINGFACE_TOKEN = "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"

# Caricamento tokenizer e modello
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # Usa float32 per CPU (o float16 per GPU)
    device_map='auto',
    token=HUGGINGFACE_TOKEN  # Autenticazione con il token
)

# Creazione del pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


# Funzione per generare una risposta
def get_response(input):

    # input_ids = tokenizer.encode(
    #     input,
    #     return_tensors='pt'
    # )

    # # Maschera di attention che fa in modo che solo i token reali vengano considerati (token di padding scartati)
    # attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    sequences = pipeline(
        input,
        do_sample=True,
        top_p=0.5,
        temperature=0.7,  # Controlla la casualità
        top_k=40,  # Considera solo i 40 token più probabili
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens= 300,
        # truncation = True
    )

    print(sequences)

    return [seq['generated_text'] for seq in sequences]



def generate_precise_prompt(edus, relationships, missing_edu):
    # Identifica tutte le relazioni che coinvolgono il nodo `missing_edu`
    related_relationships = [
        rel for rel in relationships if rel[0] == missing_edu or rel[1] == missing_edu
    ]
    
    # Trova tutti i nodi coinvolti nel sottografo relativo
    related_edus_ids = set(
        edu_id for rel in related_relationships for edu_id in (rel[0], rel[1])
    )
    
    # Filtra le EDUs e le relazioni basandosi sui nodi correlati
    subgraph_edus = [edu for edu in edus if edu[0] in related_edus_ids]
    subgraph_relationships = [
        rel for rel in relationships if rel[0] in related_edus_ids and rel[1] in related_edus_ids
    ]
    
    # Mappa vecchi indici delle EDUs in nuovi indici
    edu_id_map = {edu[0]: idx for idx, edu in enumerate(subgraph_edus)}
    
    # Rimappa le relazioni con i nuovi indici
    remapped_relationships = [
        (edu_id_map[rel[0]], edu_id_map[rel[1]], rel[2]) for rel in subgraph_relationships
    ]

    # Rimappa l'indice di `missing_edu`
    remapped_missing_edu = edu_id_map[missing_edu]

    # Crea il testo per le EDUs del sottografo
    subgraph_edus_text = "\n".join(
        [f"EDU{i+1}: '{edu[1]}'" for i, edu in enumerate(subgraph_edus)]
    )
    
    # Crea il testo per le relazioni del sottografo
    subgraph_relationships_text = "\n".join(
        [f"[EDU{rel[0]+1}] -> [EDU{rel[1]+1}]: {rel[2]}" for rel in remapped_relationships]
    )

    print(subgraph_edus)
    print(subgraph_edus_text)
    print(subgraph_relationships)
    print(subgraph_relationships_text)

    relationship_types_text = """
    ### Relationship Types:
    1. Question_answer_pair (QAP): One EDU poses a question, and the other provides an answer.
    2. Comment: One EDU adds an observation or remark related to the other.
    3. Acknowledgement: One EDU acknowledges or affirms the information in the other.
    4. Continuation: One EDU continues the idea or topic introduced in the other.
    5. Elaboration: One EDU expands on or provides more detail about the other.
    6. Q_Elab (Q-Elab): One EDU elaborates on the question posed by the other.
    7. Contrast: One EDU presents information that contrasts with the other.
    8. Explanation: One EDU provides an explanation or reasoning related to the other.
    9. Clarification_question: One EDU seeks clarification about the information in the other.
    10. Result: One EDU describes the outcome or result of the situation in the other.
    11. Correction: One EDU corrects or refines the information presented in the other.
    12. Parallel: Both EDUs describe similar or parallel ideas.
    13. Conditional: One EDU describes a condition necessary for the other to occur.
    14. Alternation: One EDU presents an alternative to the other.
    15. Background: One EDU provides background information relevant to the other.
    16. Narration: One EDU describes a sequence of events or tells a story.
    17. Confirmation_question: One EDU seeks confirmation or validation of the information in the other.
    18. Sequence: One EDU describes an event that occurs before or after the other.
    """


    # Prompt dettagliato
    prompt = f"""
    You are a language model trained to analyze and generate discourse units (EDUs).
    Your task is to ensure that the semantic relationships between EDUs in a graph are preserved,
    even when one EDU is removed. When one EDU is removed, you must generate a new EDU that replaces
    the removed one while maintaining all original relationships in the graph.

    {relationship_types_text}

    ### Rules:
    1. Each EDU represents a single, coherent idea.
    2. Relationships between EDUs are defined in the format: [EDU1] -> [EDU2]: <Relation>.
    3. When an EDU is removed:
      - Generate a new EDU that logically replaces the removed EDU.
      - Ensure the generated EDU fits naturally with all connected EDUs.
      - Preserve all semantic relationships involving the missing EDU.

    ### Example:
    Subgraph EDUs:
    EDU1: 'The sky is overcast.'
    EDU2: 'It might rain later.'

    Subgraph Relationships:
    [EDU1] -> [EDU2]: Cause-Effect.

    If EDU1 is removed:
    Generate a new EDU that logically replaces EDU1: 'The weather looks gloomy, which might indicate rain.'

    If EDU2 is removed:
    Generate a new EDU that logically replaces EDU2: 'The overcast sky suggests possible precipitation.'

    ---

    Now, process the following graph:

    Subgraph EDUs:
    {subgraph_edus_text}

    Subgraph Relationships:
    {subgraph_relationships_text}

    If EDU{remapped_missing_edu+1} is removed:
    Generate a new EDU that logically replaces EDU{remapped_missing_edu+1}:

    """

    # a new EDU to replace EDU{remapped_missing_edu+1}. The new EDU must maintain all the original semantic
    # relationships of EDU{remapped_missing_edu+1} with the remaining EDUs:

    return prompt, remapped_missing_edu



# Funzione per estrarre l'EDU generata per un'EDU specifica rimossa
def extract_generated_edu(response, removed_edu):
    # Pattern per catturare solo la parte dopo "Now, process the following input"
    main_section_pattern = r"Now, process the following input:(.*)"

    for text in response:
        # Trova solo la parte rilevante del prompt
        main_section_match = re.search(main_section_pattern, text, re.DOTALL)
        if main_section_match:
            main_section = main_section_match.group(1)

            # Cerca il testo generato per la EDU mancante
            edu_pattern = rf"If {removed_edu} is removed:\s*Generate:\s*(.*?)(?=\n\n)"
            edu_match = re.search(edu_pattern, main_section, re.DOTALL)

            if edu_match:
                return edu_match.group(1).strip()

    return None


if __name__ == '__main__':

    dataset_name_list = ["STAC_training"]

    for dataset_name in dataset_name_list:
        file_path = get_filepath(dataset_name)
    
        data = load_data(file_path)

        # Farlo per ogni dialogo del dataset
        graph = crea_grafo_da_json([data[0]])

        # Aggiungere metodo di scelta nodo da sostituire
        target_node = random.choice(list(graph.nodes))

        print(target_node)

        subgraph = get_subgraph(target_node, graph)

        edus_list = []
        relations_list = []

        for node, data in subgraph.nodes(data=True):
            edus_list.append([node, data.get('text')])

        
        for source, target, type in subgraph.edges(data=True):  # Include gli attributi dell'arco
            relations_list.append((source, target, type.get('relationship')))

        print(edus_list)
        print(relations_list)

        missing_edu = target_node

        prompt, remapped_missing_edu = generate_precise_prompt(edus_list, relations_list, missing_edu)

        print(prompt)

        try:
            print('Producing response...')
            response = get_response(prompt)
            print("Pipeline executed successfully.")
            print(response)
            generated_edu = extract_generated_edu(response, remapped_missing_edu+1)
            print(generated_edu)
        except Exception as e:
            print("Error occurred:")
            print(e)


    # # prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
    # edus = ["how do i undo the things that easy ubuntu has done ? : o", "for security upgrades and the like ?"]
    # relationships = [{"source": "EDU1", "target": "EDU2", "relation": "Clarification_question"}]
    # missing_edu = "EDU1"

    # prompt = generate_precise_prompt(edus, relationships, missing_edu)

    # print(prompt)

    # print('Producing response...')
    # # response = get_response(prompt)
    # # print(response)

    # try:
    #     response = get_response(prompt)
    #     print("Pipeline executed successfully.")
    #     print(response)
    #     generated_edu = extract_generated_edu(response, missing_edu)
    #     print(generated_edu)
    # except Exception as e:
    #     print("Error occurred:")
    #     print(e)