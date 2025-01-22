from openai import OpenAI
import torch
import re
from scripts.get_edu import *
from scripts.vincoli_edu import *
import random
from scripts.MPNet_embedding import *

client = OpenAI(api_key="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")

# Usando l'API gpt4-all
# client = OpenAI(api_key="g4a-5KHHrU4Ow3zoD3kPl1O8TJjNjjoh5aWTAid", base_url="https://api.gpt4-all.xyz/v1")


def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        n = 3
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
    2. Comment: One EDU adds an observation or remark related to the other.
    3. Acknowledgement: One EDU acknowledges or affirms the information in the other.
    4. Continuation: One EDU continues the idea or topic introduced in the other.
    5. Elaboration: One EDU expands on or provides more detail about the other.
    6. Q_Elab: One EDU elaborates on the question posed by the other.
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

    print(relationships_prompt)

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


def get_new_edu(data, dialogue_index, dataset_name):
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

    print(prompt)

    try:
        # print('Producing response...')
        response = get_response(prompt)
        # print("Pipeline executed successfully.")
    except Exception as e:
        print("Error occurred:")
        print(e)

    for i, choice in enumerate(response.choices):
        print(f"Risposta {i+1}:")
        print(choice.message.content)

    output = response.choices[0].message.content
    # Rimuove solo i caratteri ' all'inizio e alla fine
    new_edu = output.strip("'")

    print(f'Old EDU: {data["text"]}')
    print(f'New EDU: {new_edu}')

    return new_edu, target_node


def get_new_edu_emb(new_edu):
    return create_one_embedding(new_edu)


if __name__ == '__main__':
    dataset_name_list = ["STAC_training"]

    for dataset_name in dataset_name_list:
        file_path = get_filepath(dataset_name)
    
        data = load_data(file_path)

        # for i, dialogue in enumerate(data):
            # graph = crea_grafo_da_json([dialogue])

        dialogue_index = 0
        new_edu = get_new_edu(data, dialogue_index, dataset_name)
        embedding_new_edu = create_one_embedding(new_edu)