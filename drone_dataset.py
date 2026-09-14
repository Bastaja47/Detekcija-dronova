                                                                                                                                       

"""
drone_dataset.py

Deljeni modul koji sadrži Dataset klasu i pomoćne funkcije za učitavanje
drone RF dataset-a. Uvozimo ga u svaki notebook/skript kojem treba pristup
podacima, umesto da kod kopiramo svuda.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DroneRFDataset(Dataset):
    """
    Custom PyTorch Dataset za drone RF signal dataset.
    Koristi spektrograme (x_spec) kao ulaz, i klasu (y) kao labelu.
    """

    def __init__(self, data_dict, indices):
        self.x_spec = data_dict['x_spec']
        self.y = data_dict['y']
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        spectrogram = self.x_spec[real_idx]
        label = self.y[real_idx]
        return spectrogram, label


def load_data_and_splits(data_path='dataset.pt', split_path='train_val_test_split.npz'):
    """
    Učitava ceo dataset i sačuvanu train/val/test podelu.
    Vraća: data (dict), train_indices, val_indices, test_indices
    """
    data = torch.load(data_path, weights_only=False)
    split = np.load(split_path)
    return data, split['train_indices'], split['val_indices'], split['test_indices']


def get_dataloaders(batch_size=64, num_workers=0):
    """
    Sve-u-jednom funkcija: učitava podatke, pravi Dataset i DataLoader objekte
    za train/val/test, i vraća ih spremne za korišćenje.
    """
    data, train_idx, val_idx, test_idx = load_data_and_splits()

    train_dataset = DroneRFDataset(data, train_idx)
    val_dataset = DroneRFDataset(data, val_idx)
    test_dataset = DroneRFDataset(data, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader