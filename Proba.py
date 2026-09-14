import torch
checkpoint = torch.load("dataset.pt")
print(checkpoint.keys())