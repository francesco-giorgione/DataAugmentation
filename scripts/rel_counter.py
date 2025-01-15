import json
from collections import defaultdict
import matplotlib.pyplot as plt

def conta_relazioni(json_file, output_file, path_file):
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

        tmp_str = []
        # Stampa i risultati con percentuali
        for tipo_relazione, conteggio, percentuale in zip(tipi_relazioni, conteggi, percentuali):
            tmp_str.append(f"{tipo_relazione}: {conteggio} ({percentuale:.2f}%)")
            # print(tmp_str)

        with open(path_file, "w") as file:
            for el in tmp_str:
                file.write(el + "\n")

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

        # Estendi i limiti dell'asse y per lasciare spazio ai valori sopra le barre
        plt.ylim(0, max(percentuali) + 5)

        # Aggiungi margini per evitare tagli
        plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)

        # Salva l'istogramma su un file
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Grafico salvato come {output_file}")
        plt.close()
    else:
        print("Non ci sono relazioni nel file JSON.")


if __name__ == '__main__':
    # Specifica il nome del file JSON di input e il file di output per l'istogramma
    json_input_file = 'STAC_testing.json'  # Modifica con il nome del tuo file JSON
    output_image_file = 'relazioni_istogramma.png'  # Modifica con il nome del file di output desiderato


    conta_relazioni('dataset/STAC/train_subindex.json', 'graphic/STAC_relazioni_istogramma.png', 'info_rel/STAC_rel.txt')
    conta_relazioni('dataset/MOLWENI/train.json', 'graphic/MOLWENI_relazioni_istogramma.png', 'info_rel/MOLWENI_rel.txt')
    conta_relazioni('dataset/MINECRAFT/TRAIN_307_bert.json', 'graphic/MINECRAFT_relazioni_istogramma.png', 'info_rel/MINECRAFT_rel.txt')
