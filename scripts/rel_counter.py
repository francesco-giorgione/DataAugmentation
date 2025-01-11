import json
from collections import defaultdict
import matplotlib.pyplot as plt

def conta_relazioni(json_file, output_file):
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

    # Se ci sono relazioni, calcola percentuali e plottale
    if totale_relazioni > 0:
        relazioni_ordinate = sorted(contatore_relazioni.items(), key=lambda x: x[1], reverse=True)

        # Estrai i dati per il grafico
        tipi_relazioni = [tipo_relazione for tipo_relazione, _ in relazioni_ordinate]
        conteggi = [conteggio for _, conteggio in relazioni_ordinate]
        percentuali = [(conteggio / totale_relazioni) * 100 for conteggio in conteggi]

        # Stampa i risultati con percentuali
        for tipo_relazione, conteggio, percentuale in zip(tipi_relazioni, conteggi, percentuali):
            print(f"{tipo_relazione}: {conteggio} ({percentuale:.2f}%)")

        # Genera l'istogramma
        plt.figure(figsize=(10, 6))
        bars = plt.bar(tipi_relazioni, percentuali, color='skyblue')

        # Aggiungi i contatori sopra ogni barra
        for bar, conteggio in zip(bars, conteggi):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(conteggio),
                     ha='center', va='bottom', fontsize=10, color='black')

        plt.xlabel('Tipi di Relazioni')
        plt.ylabel('Percentuale (%)')
        plt.title('Distribuzione delle Relazioni')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Salva l'istogramma su un file
        plt.savefig(output_file)
        print(f"Grafico salvato come {output_file}")
        plt.close()
    else:
        print("Non ci sono relazioni nel file JSON.")


if __name__ == '__main__':
    # Specifica il nome del file JSON di input e il file di output per l'istogramma
    json_input_file = 'STAC_testing.json'  # Modifica con il nome del tuo file JSON
    output_image_file = 'relazioni_istogramma.png'  # Modifica con il nome del file di output desiderato
    conta_relazioni(json_input_file, output_image_file)
