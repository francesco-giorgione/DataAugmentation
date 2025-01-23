from openai import OpenAI
import torch
import re
from get_edu import *
from vincoli_edu import *
import random
from MPNet_embedding import *

# client = OpenAI(api_key="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")

# Usando l'API gpt4-all
# client = OpenAI(api_key="g4a-5KHHrU4Ow3zoD3kPl1O8TJjNjjoh5aWTAid", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-IeOj4y3qEhbNFblgYpwhcUOHImM1DStIi6L", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-eIS6lO1ZHWs7XCD24lTxR5k5dWrsNsEai6y", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-5UFK4uftVxD857EJ3NSdAXWIMedLTU9Stnh", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-aSzdXjOFgGlg64Z7i1CjCL5qXHF3WP72oOR", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-v46sGFCdhj6d5AP6pQSmDTszAAcmNxDpSkr", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-dNBfm3lK4ZHABQmGOmLxNVbMBeBvIp5DZzV", base_url="https://api.gpt4-all.xyz/v1")
client = OpenAI(api_key="g4a-bKeS8JvQiSrnUFyIKKJrYcZNHbuDvNv75pv", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-wv93UpiCmIsQ6bx4GSDFdsfXysCHevSnegf", base_url="https://api.gpt4-all.xyz/v1")
# client = OpenAI(api_key="g4a-efCvQW5V9nF7UiDJo1Usu4i3tSQtgnTdauT", base_url="https://api.gpt4-all.xyz/v1")


def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        # n = 3
    )

    return response



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
        rel for rel in related_relationships if rel[0] == missing_edu or rel[1] == missing_edu
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

    # print(subgraph_edus)
    # print(subgraph_edus_text)
    # print(subgraph_relationships)
    # print(subgraph_relationships_text)

    relationship_types_text = """
    ### Relationship Types:
    1. Question_answer_pair: One EDU poses a question, and the other provides an answer.
    2. Comment: One EDU adds an opinion or evaluation of the content of the other.
    3. Acknowledgement: One EDU acknowledges or affirms the information in the other, signaled by words like OK, Right, Right then, Good, Fine.
    4. Continuation: Both EDUs elaborate or provide background to the same segment.
    5. Elaboration: One EDU provides more information about the eventuality introduced in the other.
    6. Q_Elab: One EDU elaborates on the question posed by the other.
    7. Contrast: Both EDUs have similar semantic structures, but present information that contrast each other.
    8. Explanation: One EDU provides an explanation or reasoning related to the other.
    9. Clarification_question: One EDU seeks clarification about the information in the other.
    10. Result: One EDU describes the cause of its effect described in the other.
    11. Correction: One EDU corrects or refines the information presented in the other.
    12. Parallel: Both EDUs describe similar or parallel ideas.
    13. Conditional: One EDU describes a condition necessary for the other to occur.
    14. Alternation: One EDU presents an alternative to the other.
    15. Background: One EDU provides background information relevant to the other.
    16. Narration: One EDU describes a sequence of events or tells a story.
    17. Confirmation_question: One EDU seeks confirmation or validation of the information in the other.
    18. Sequence: One EDU describes an event that occurs before or after the other.
    19. Question-answer_pair: One EDU poses a question, and the other provides an answer.
    20. QAP: One EDU poses a question, and the other provides an answer.
    21. Q-Elab: One EDU elaborates on the question posed by the other.
    """

    relationship_dict = {
    match.group(1): match.group(0)
    for match in re.finditer(r'(\w+):.*', relationship_types_text)
    }

    relationships_prompt = "### Relationship Types: \n"

    i = 1

    for rel in subgraph_relationships:
        relation_type = rel[2]
        if relation_type in relationship_dict:
            relationships_prompt += f"{i}. " + relationship_dict[relation_type] + "\n"
            i += 1
        else:
            print(f"La relazione '{relation_type}' NON è valida.")

    # print(relationships_prompt)

    # Prompt dettagliato
    prompt = f"""
    When one EDU is removed, you must generate a new EDU that replaces the removed one. The new EDU must be a strict 
    paraphrase of the removed EDU, without referencing, modifying, or inferring any details from the other EDUs. The
    relationships between EDUs will remain implicitly preserved without needing to be explicitly mentioned in the generated EDU.

    ### Rules:
    1. Each EDU represents a single, coherent idea.
    2. Relationships between EDUs are defined in the format: [EDU1] -> [EDU2]: <Relation>.
    3. When an EDU is removed:
      - Generate a new EDU that is strictly a paraphrase of the removed EDU. 
        The new EDU must only contain information present in the removed EDU 
        and must not introduce, modify, or infer any details from other EDUs 
        in the graph. Do not include any content from EDUs before or after it.
      - Ensure the generated EDU fits naturally with all connected EDUs.
      - Preserve all semantic relationships involving the missing EDU.

    {relationships_prompt}

    ### Example:
    Input EDUs:
    EDU1: 'how is the sky?'
    EDU2: 'The sky is overcast.'
    EDU3: 'It might rain later.'

    Relationships:
    [EDU1] -> [EDU2]: Question_answer_pair
    [EDU2] -> [EDU3]: Comment.

    The removed EDU is the EDU that appears in all relations specified in relationships

    EDU2 is removed. Generate a new EDU that logically replaces only EDU2, preserving all semantic
    relationships of EDU2 with other EDUs, defined in the previous list Relationships: 
    'The weather looks gloomy, which might indicate rain.'

    ---

    Now, process the following graph:

    Input EDUs:
    {subgraph_edus_text}

    Relationships:
    {subgraph_relationships_text}

    EDU{remapped_missing_edu+1} is removed.
    Generate a new EDU that logically replaces only EDU{remapped_missing_edu+1}, preserving all semantic
    relationships of EDU{remapped_missing_edu+1} with other EDUs, defined in the previous list Relationships:

    """

    return prompt


def get_new_edus(data, dialogue_index, dataset_name):
    graph = crea_grafo_da_json([data[dialogue_index]])
    target_node = get_edu_from_DAG(
        dataset_name,
        graph)[0]

    print(f'Chosen node {target_node} in dialogue {dialogue_index}')
    subgraph = get_subgraph(target_node, graph)

    edus_list = []
    relations_list = []

    for node, data in subgraph.nodes(data=True):
        edus_list.append([node, data.get('text')])

    for source, target, type in subgraph.edges(data=True):  # Include gli attributi dell'arco
        relations_list.append((source, target, type.get('relationship')))

    missing_edu = target_node
    prompt = generate_precise_prompt(edus_list, relations_list, missing_edu)


    try:
        # print('Producing response...')
        response = get_response(prompt)
        # print("Pipeline executed successfully.")
    except Exception as e:
        print("Error occurred:")
        print(e)

    new_edus = [c.message.content.strip("'") for c in response.choices]

    print(f'Old EDU: {data["text"]}')
    for i, edu in enumerate(new_edus, start=1):
        print(f'New EDU {i}: {edu}')

    return new_edus, target_node


def get_new_edus_gpt4all(data, dialogue_index, dataset_name):
    graph = crea_grafo_da_json([data[dialogue_index]])
    target_node = get_edu_from_DAG(
        dataset_name,
        graph)[0]

    print(f'Chosen node {target_node} in dialogue {dialogue_index}')
    subgraph = get_subgraph(target_node, graph)

    edus_list = []
    relations_list = []

    for node, data in subgraph.nodes(data=True):
        edus_list.append([node, data.get('text')])

    for source, target, type in subgraph.edges(data=True):  # Include gli attributi dell'arco
        relations_list.append((source, target, type.get('relationship')))

    missing_edu = target_node
    prompt = generate_precise_prompt(edus_list, relations_list, missing_edu)

    # Numero di risposte da generare
    n = 3
    responses_list = []

    try:
        # print('Producing response...')
        for i in range(n):
            response = get_response(prompt)
            responses_list.append(response)
            
    except Exception as e:
        print("Error occurred:")
        print(e)

    # for i, choice in enumerate(response.choices):
    #     print(f"Risposta {i+1}:")
    #     print(choice.message.content)
    #
    # output = response.choices[0].message.content
    # # Rimuove solo i caratteri ' all'inizio e alla fine
    # new_edu = output.strip("'")
    new_edus = []

    print(f'Old EDU: {data["text"]}')

    for response in responses_list:
        
        new_edu = response.choices[0].message.content.strip("'")
        new_edus.append(new_edu)

        for i, edu in enumerate(new_edus, start=1):
            print(f'New EDU {i}: {edu}')

    return new_edus, target_node


def get_new_edus_gpt4all(data, dialogue_index, dataset_name):
    graph = crea_grafo_da_json([data[dialogue_index]])
    target_node = get_edu_from_DAG(
        dataset_name,
        graph)[0]

    print(f'Chosen node {target_node} in dialogue {dialogue_index}')
    subgraph = get_subgraph(target_node, graph)

    edus_list = []
    relations_list = []

    for node, data in subgraph.nodes(data=True):
        edus_list.append([node, data.get('text')])

    for source, target, type in subgraph.edges(data=True):  # Include gli attributi dell'arco
        relations_list.append((source, target, type.get('relationship')))

    missing_edu = target_node
    prompt = generate_precise_prompt(edus_list, relations_list, missing_edu)

    # Numero di risposte da generare
    n = 3
    responses_list = []

    try:
        # print('Producing response...')
        for i in range(n):
            response = get_response(prompt)
            responses_list.append(response)
            
    except Exception as e:
        print("Error occurred:")
        print(e)

    # for i, choice in enumerate(response.choices):
    #     print(f"Risposta {i+1}:")
    #     print(choice.message.content)
    #
    # output = response.choices[0].message.content
    # # Rimuove solo i caratteri ' all'inizio e alla fine
    # new_edu = output.strip("'")
    new_edus = []

    print(f'Old EDU: {data["text"]}')

    for response in responses_list:
        
        new_edu = response.choices[0].message.content.strip("'")
        new_edus.append(new_edu)

        for i, edu in enumerate(new_edus, start=1):
            print(f'New EDU {i}: {edu}')

    return new_edus, target_node


def get_new_edus_emb(new_edus):
    return [create_one_embedding(e) for e in new_edus]


if __name__ == '__main__':
    dataset_name_list = ["STAC_training"]

    for dataset_name in dataset_name_list:
        file_path = get_filepath(dataset_name)
    
        data = load_data(file_path)

        # for i, dialogue in enumerate(data):
            # graph = crea_grafo_da_json([dialogue])

        dialogue_index = 0
        new_edu = get_new_edus_gpt4all(data, dialogue_index, dataset_name)
        embedding_new_edu = create_one_embedding(new_edu)