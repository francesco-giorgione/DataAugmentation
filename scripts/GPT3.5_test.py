import openai

# Inserisci la tua chiave API
openai.api_key = "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"

# Funzione per inviare una richiesta al modello GPT-3.5
def usa_gpt_3_5(prompt, max_tokens=150):
    try:
        # Chiamata al modello GPT-3.5
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Specifica il modello da usare
            messages=[
                {"role": "system", "content": "Sei un assistente esperto."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,  # Numero massimo di token di output
            temperature=0.7,  # Creatività del modello
        )
        # Estrarre la risposta generata
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Errore: {e}"


if __name__ == '__main__':
    prompt = "Spiegami il concetto di machine learning in modo semplice."
    risposta = usa_gpt_3_5(prompt)
    print(risposta)
