import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

    # Genera una continuazione del testo
    output = model.generate(
        input_ids,
        max_length=input_ids.size(1) + 30,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.2,
        temperature=0.5,
        do_sample=True,
        attention_mask=attention_mask
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


    response = get_response(prompt_2)


    print(f'{prompt_2}\n\n{response}')