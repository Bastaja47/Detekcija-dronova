"""
test_api.py

Test skripta - uzima STVARNE uzorke iz TEST skupa (model ih nikad nije video)
i šalje ih API-ju, da proverimo ponašanje na raznovrsnim, nepristrasnim primerima.
"""

import requests
import torch
import numpy as np

data = torch.load('dataset.pt', weights_only=False)
split = np.load('train_val_test_split.npz')
test_indices = split['test_indices']

class_names = ['DJI', 'FutabaT14', 'FutabaT7', 'Graupner', 'Noise', 'Taranis', 'Turnigy']

                                                                          
print(f"{'Stvarna klasa':<12} {'Predikcija':<12} {'Confidence':<12} {'SNR':<6} {'Tačno?'}")
print("-" * 60)

for class_id in range(7):
                                                             
    class_mask = data['y'][test_indices] == class_id
    class_test_indices = test_indices[class_mask.numpy()]

                               
    sample_idx = class_test_indices[0]

    spectrogram = data['x_spec'][sample_idx]
    true_label = data['y'][sample_idx].item()
    snr = data['snr'][sample_idx].item()

    spectrogram_list = spectrogram.tolist()

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"data": spectrogram_list}
    )

    result = response.json()
    predicted = result['predicted_class']
    confidence = result['confidence']
    correct = "✓" if predicted == class_names[true_label] else "✗"

    print(f"{class_names[true_label]:<12} {predicted:<12} {confidence:<12} {snr:<6} {correct}")