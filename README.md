# Data Augmentation di strutture di dialoghi Multi-Party con Generazione Vincolata e Validazione GNN

Questo progetto si propone di sviluppare una pipeline per la Data Augmentation
di strutture di dialoghi multi-parte, sfruttando modelli di tipo Large Language Model
(LLM). Si vogliono generare nuovi dialoghi sintetici che rispettino rigorosamente 
i vincoli strutturali e semantici individuati sui dialoghi esistenti. La validazione dei 
dialoghi generati è implementata come un task di link prediction, eseguito tramite
Graph Neural Network (GNN). L'obiettivo è di verificare che, per ogni nuovo dialogo, le
relazioni tra unità del discorso (EDUs) siano preservate. I dialoghi che superano il processo
di validazione vengono inseriti nel dataset aumentato. Infine, il task di _Dialogue Discourse Parsing as Generation_
(vedi https://github.com/chuyuanli/Seq2Seq-DDP)
viene eseguito sul dataset di partenza e su quello aumentato, in modo da confrontare i risultati ottenuti
e verificare se l'augmentation ha determinato un miglioramento delle performance.










## Requisiti
L'esecuzione del progetto richiede l'installazione delle dipendenze, da eseguirsi tramite il comando
```
    pip install requirements.txt
```


## Autori e contatti
| Autore              | Indirizzo email                |
|---------------------|--------------------------------|
| Simona Lo Conte     | s.loconte2@studenti.unisa.it   |
| Marta Napolillo     | m.napolillo1@studenti.unisa.it |
| Francesco Giorgione | f.giorgione4@studenti.unisa.it |

