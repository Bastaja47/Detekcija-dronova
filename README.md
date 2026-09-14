# Detekcija dronova putem RF signala

Projekat iz predmeta Tehnologije i alati u mašinskom učenju. Cilj je istrenirati model koji na osnovu radio-frekvencijskog (RF) signala prepoznaje tip drona i razlikuje ga od šuma, uz kompletan MLOps tok: treniranje → AutoML → evaluacija → REST API serviranje.

## Dataset

[Noisy Drone RF Signal Classification](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification) (Kaggle) — 98.705 označenih uzoraka, 7 klasa (6 tipova dronova + Noise), u obliku sirovih IQ signala i gotovih spektrograma.

> `dataset.pt` **nije uključen u ovaj repozitorijum** (fajl je prevelik za Git)

| Klasa | Broj uzoraka |
|---|---|
| DJI | 2.194 |
| FutabaT14 | 6.938 |
| FutabaT7 | 3.661 |
| Graupner | 6.481 |
| Noise | 52.552 |
| Taranis | 16.546 |
| Turnigy | 10.333 |

## Struktura projekta

```
├── dataset.pt                      # preuzeti sa Kaggle-a (vidi gore)
├── class_stats.csv                 # deo originalnog dataset-a
├── SNR_stats.csv                   # deo originalnog dataset-a
├── train_val_test_split.npz        # sačuvana stratifikovana podela (70/15/15, seed 42)
│
├── drone_dataset.py                # PyTorch Dataset klasa + učitavanje podataka
├── train_utils.py                  # trening petlja sa early stopping-om
├── models.py                       # ConfigurableCNN arhitektura
│
├── 01_explore_dataset.ipynb        # istraživanje dataset-a, stratifikovan split
├── 02_data_pipeline.ipynb          # PyTorch Dataset/DataLoader
├── 03_baseline_model.ipynb         # ručno dizajniran CNN, trening, evaluacija
├── 04_automl.ipynb                 # Optuna pretraga hiperparametara
├── 05_evaluation.ipynb             # finalno poređenje baseline vs AutoML
│
├── baseline_model.pth              # sačuvane težine baseline modela
├── automl_model.pth                # sačuvane težine AutoML modela
├── automl_best_params.json         # hiperparametri koje je Optuna pronašla
│
├── api.py                          # FastAPI REST server
├── test_api.py                     # testna skripta za API
│
├── toggle_comments.py              # pomoćna skripta (vidi ispod)
├── requirements.txt
└── .gitignore
```

## Instalacija

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Pokretanje

Notebook-ovi se pokreću redom (svaki zavisi od fajlova koje prethodni sačuva):

1. `01_explore_dataset.ipynb` — pravi `train_val_test_split.npz`
2. `02_data_pipeline.ipynb` — Dataset/DataLoader provera
3. `03_baseline_model.ipynb` — trenira i čuva `baseline_model.pth`
4. `04_automl.ipynb` — Optuna pretraga, čuva `automl_model.pth` i `automl_best_params.json`
5. `05_evaluation.ipynb` — finalno poređenje modela

Zatim, za REST API:

```bash
uvicorn api:app --reload
```

U drugom terminalu (dok server radi):

```bash
python test_api.py
```

API dokumentacija (interaktivna): `http://127.0.0.1:8000/docs`

## MLOps komponente

| Komponenta | Status |
|---|---|
| Treniranje i evaluacija (uz AutoML) | Praktično implementirano |
| Primena i serviranje modela (REST API) | Praktično implementirano |
| Monitoring | Teorijski pokriveno |

## Rezultati

| Metrika | Baseline | AutoML |
|---|---|---|
| Accuracy | 93.16% | **94.06%** |
| Macro F1 | 88.94% | **90.74%** |
| Macro Recall | 84.21% | **87.21%** |
| DJI Recall (najteža klasa) | 69.00% | **77.81%** |

AutoML model (Optuna, 20 trial-ova) je pronašao arhitekturu sa 4 konvoluciona sloja, 32 bazna filtera, dropout 0.222, learning rate 0.0007. Poboljšanje u odnosu na baseline je najizraženije kod retkih klasa i pri niskom SNR-u (zašumljeni signali) — detaljna analiza u `05_evaluation.ipynb`.



## Autor

Projekat izrađen u okviru predmeta Tehnologije i alati u mašinskom učenju.
