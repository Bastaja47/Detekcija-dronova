"""
api.py

REST API za serviranje drone RF klasifikacionog modela.
Pokreće se komandom: uvicorn api:app --reload
"""

import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from models import ConfigurableCNN

app = FastAPI(title="Drone RF Detection API")

                                                                     
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('automl_best_params.json', 'r') as f:
    best_params = json.load(f)

model = ConfigurableCNN(
    num_conv_layers=best_params['num_conv_layers'],
    base_filters=best_params['base_filters'],
    dropout_rate=best_params['dropout_rate']
).to(device)
model.load_state_dict(torch.load('automl_model.pth', weights_only=True, map_location=device))
model.eval()                                                   

CLASS_NAMES = ['DJI', 'FutabaT14', 'FutabaT7', 'Graupner', 'Noise', 'Taranis', 'Turnigy']

print(f"Model učitan na uređaj: {device}")


                                                                    
class SpectrogramInput(BaseModel):
    """
    Klijent mora poslati 'data' polje: 3D lista brojeva oblika [2, 128, 128]
    (isti oblik kao jedan uzorak iz x_spec u našem dataset-u).
    """
    data: List[List[List[float]]]


@app.get("/")
def root():
    return {"poruka": "Drone RF Detection API radi!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: SpectrogramInput):
    """
    Prima spektrogram [2, 128, 128], vraća predviđenu klasu i verovatnoće
    za sve klase.
    """
                                                           
    spectrogram = torch.tensor(input_data.data, dtype=torch.float32)

                                                                             
                                                             
    spectrogram = spectrogram.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(spectrogram)                                

                                                                             
        probabilities = F.softmax(outputs, dim=1)[0]                                      

        predicted_idx = torch.argmax(probabilities).item()

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item(), 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(probabilities[i].item(), 4)
            for i in range(len(CLASS_NAMES))
        }
    }