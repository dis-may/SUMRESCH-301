import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dsms_predictor_nn import dSMSPredictorNN
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum, auto

from vehicle_functions import *

BASE_NAME = "love"
# BASE_NAME = "aggression"
# BASE_NAME = "fear"
# BASE_NAME = "explorer"

class VehicleType(Enum):
    AGGRESSION = auto()
    FEAR = auto()
    LOVE = auto()
    EXPLORER = auto()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

errs = []


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += (pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
    test_loss /= num_batches
    correct /= size
    errs.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")      

# Import bs_data.npy to use as dataset
inputs          = torch.from_numpy(np.load(f'{BASE_NAME}_inputs.npy')).float()
correct_outputs = torch.from_numpy(np.load(f'{BASE_NAME}_correct_outputs.npy')).float()

# 2. Create a TensorDataset
dataset = TensorDataset(inputs,correct_outputs)

# 3. Create a DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=256*4,
    shuffle=True, # Shuffles data every epoch
    # num_workers=8 # Use multiple processes for data loading # needs to be commented out for my Mac to run it lol
)

loss_fn = nn.MSELoss()#CrossEntropyLoss()


model = get_model(VehicleType.AGGRESSION)
test(dataloader, model, loss_fn)

print(errs)