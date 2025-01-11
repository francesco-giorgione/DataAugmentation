import torch

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(
        **batch,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        top_p=0.7,
        temperature=2.5,
        do_sample=True
    )
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


if __name__ == '__main__':
    sentences_1 = [
        "TEXT: 'I feel bad today'\n" +
        "INSTRUCTION: Rewrite TEXT in a more creative way.\n" +
        "EXAMPLE: 'It is raining outside' → 'The skies are pouring their tears today.'\n"
    ]

    sentences_2 = [
        "Which course should I take to get started in data science?",
        "She wasn't looking for a knight, she was looking for a sword.",
        "even with installing it …em to be able to use it",
        "perhaps it should be bet…ng with webserver ? : -",
        "In the end, we only regret the chances we didn't take.",
        "I dreamt I am running on sand in the night",
        "Long long ago, there lived a king and a queen. For a long time, they had no children.",
        "I am typing the best article on paraphrasing with Transformers.",
        "I feel happy today, and you?"
    ]

    for sentence in sentences_1:
        # Generate paraphrased sentence
        result = get_response(sentence, num_return_sequences=3, num_beams=10)

        print(f"Original: {sentence}")
        for i, r in enumerate(result, start=1):
            print(f"Paraphrase {i}: {r}")
        print('\n')
