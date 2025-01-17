from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

HUGGINGFACE_TOKEN = "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"
tokenizer = AutoTokenizer.from_pretrained(model, token=HUGGINGFACE_TOKEN)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,  # Usa float32 per CPU (o float16 per GPU)
    device_map='auto',
    token=HUGGINGFACE_TOKEN  # Autenticazione con il token
)


def get_response(input):
    sequences = pipeline(
        input,
        do_sample=True,
        top_p=0.5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )

    return [seq['generated_text'] for seq in sequences]


if __name__ == '__main__':
    prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
    print('Producing response...')
    response = get_response(prompt)
    print(response)
