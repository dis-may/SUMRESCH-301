import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dsms_predictor_nn import dSMSPredictorNN
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



#BASE_NAME = "bs"
#BASE_NAME = "love"
BASE_NAME = "aggression"


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
    num_workers=8 # Use multiple processes for data loading
)

for X, y in dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# for batch in dataloader:
#     X = batch[0]
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


model = dSMSPredictorNN().to(device)
print(model)

loss_fn = nn.MSELoss()#CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #print(pred,X)
        errs.append(float(loss))

# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")            

def test(dataloader, model, loss_fn):
    model.eval()
    # test_input is two test cases where for each I'd like to see what the output of the
    # trained NN is
    test_input = np.array([[0,1,0,1],
                           [0.8664798,   3.7320678,   2.8454657,  -2.8668346]])
    test_tensor = torch.from_numpy(test_input).float().to(device)
    with torch.no_grad():
        pred = model(test_tensor)
        print(f"Test Input: \n {test_tensor.cpu().numpy()}")
        print(f"Model Output: \n {pred.cpu().numpy()}")



epochs = 10
errs = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
    #test(dataloader, model, loss_fn)
    #model.eval()
    #model(torch.randn(1,4).to(device))
print("Done!")

plt.plot(errs)
plt.yscale('log')
plt.show()

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
