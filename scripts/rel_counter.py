import json
from collections import defaultdict


def conta_relazioni(json_file):
    # Carica il file JSON
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Dizionario per contare le occorrenze di ogni tipo di relazione
    contatore_relazioni = defaultdict(int)

    # Totale delle relazioni
    totale_relazioni = 0

    # Itera su ogni sessione (oggetto) nel file JSON
    for session in data:
        # Ottieni le relazioni dalla sessione
        relazioni = session.get("relations", [])

        # Conta ogni tipo di relazione
        for relazione in relazioni:
            tipo_relazione = relazione.get("type")
            if tipo_relazione:
                contatore_relazioni[tipo_relazione] += 1
                totale_relazioni += 1

    # Se ci sono relazioni, ordina i risultati in base alla probabilità decrescente
    if totale_relazioni > 0:
        relazioni_ordinate = sorted(contatore_relazioni.items(), key=lambda x: x[1], reverse=True)

        # Calcola e stampa il risultato con percentuali, ordinato per frequenza
        for tipo_relazione, conteggio in relazioni_ordinate:
            percentuale = (conteggio / totale_relazioni) * 100
            print(f"{tipo_relazione}: {conteggio} ({percentuale:.2f}%)")
    else:
        print("Non ci sono relazioni nel file JSON.")


# Specifica il nome del file JSON di input
json_input_file = 'MINECRAFT_TEST_101_bert.json'  # Modifica con il nome del tuo file JSON
conta_relazioni(json_input_file)
