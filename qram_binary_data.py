import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pennylane as qml
import torch.utils.data as Data
from sklearn.utils import shuffle
import random
from utils import *
import itertools

# Declare Directory
name = "model/binary_data"
log_dir = f"{name}"
models_dir = f"{name}"

device = torch.device("cuda:0")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Initialize Variables
batch_size = 16
address_lines = int(input("Enter number of address and data lines: "))
data_lines = address_lines
n_layers = 4
qram_start_epoch = 0
qram_epochs = 100
qram_save_step = 10
lr = 1e-3

# Create train and test data
x = list(range(2**address_lines))
x_bin = []
for i in range(len(x)):
  bin_val = bin(x[i])[2:]
  while len(bin_val) < address_lines:
    bin_val = '0'+bin_val
  bin_val_arr = []
  for i in range(address_lines):
    bin_val_arr.append(int(bin_val[i]))
  x_bin.append(bin_val_arr)

data = []
for i in range(2**address_lines):
  data.append(random.randint(0,2**data_lines - 1))
# data = list(range(2**data_lines))
# for i in range(10):
#   data = shuffle(data)
data_bin = []
for i in range(len(data)):
  bin_val = bin(data[i])[2:]
  while len(bin_val) < data_lines:
    bin_val = '0'+bin_val
  bin_val_arr = []
  for i in range(data_lines):
    bin_val_arr.append(int(bin_val[i]))
  data_bin.append(bin_val_arr)  

for i in range(len(x_bin)):
  print(x_bin[i], data_bin[i])  

for i in range(2000//(2**address_lines)):
  for i in range(2**address_lines):
    x_bin.append(x_bin[i])
    data_bin.append(data_bin[i])

x_bin = torch.FloatTensor(x_bin)
data_bin = torch.FloatTensor(data_bin)

dataset_size = x_bin.shape[0]
dataset = Data.TensorDataset(x_bin, data_bin)
dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.85*len(x_bin)), len(x_bin)-int(0.85*len(x_bin))])
train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = Data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=False)

# Define classical and quantum devices.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
qdevice = qml.device("default.qubit", wires=address_lines)

# Define qram variational quantum circuit.
@qml.qnode(qdevice, interface='torch')
def qram(inputs, params, weights):
    qml.AngleEmbedding(features=inputs, wires=range(address_lines))
    for l in range(n_layers):
        j = 0
        for i in range(address_lines):
            qml.RY(params[l*36+j],wires=i)
            qml.RY(params[l*36+j+1],wires=(i+1)%address_lines)
            qml.CNOT(wires=[i,(i+1)%address_lines])
            qml.CRZ(params[l*36+j+2], wires=[i,(i+1)%address_lines])
            qml.PauliX(wires=(i+1)%address_lines)
            qml.CRX(params[l*36+j+3],wires=[i,(i+1)%address_lines])
            j += 4
    qml.StronglyEntanglingLayers(weights=weights, wires=range(address_lines))
    return [qml.expval(qml.PauliZ(i)) for i in range(data_lines)]

qram_weights = {"params":n_layers*36, "weights":(n_layers, address_lines, 3)}  
qram_layer = qml.qnn.TorchLayer(qram, qram_weights, init_method=torch.nn.init.normal_) 

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qram_layer = qram_layer

    def forward(self, adds):
        pred = self.qram_layer(adds)
        return pred

# Make model and optimizer.
model = Model().to(device)
loss_function = torch.nn.MSELoss()
opt_qram = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------
#  QRAM Training Start
# -------------------------
print("Starting training QRAM")

for epoch in range(qram_start_epoch, qram_epochs):
  curr_log = f"epoch {epoch+1}/{qram_epochs}\t"
  cnt = 0
  total = 0
  total_hd = 0

  losses = []
  for batch, (x_bin, data_bin) in enumerate(dataloader):
    opt_qram.zero_grad()
    pred = (model(x_bin)+1)/2
    loss = torch.mean(torch.sum((pred - data_bin)**2, 1))
    loss.backward()
    opt_qram.step()
    calculated_loss = (pred-data_bin)**2
    filter = torch.where(calculated_loss >= 0.25, True, False)
    total += x_bin.shape[0]
    total_hd += torch.sum(filter==True).item()
    for i in range(filter.shape[0]):
      if True in filter[i]:
        pass
      else:
        cnt += 1

    losses.append(loss.item())
    print(f"{epoch+1}/{batch}\t loss:{loss.item():.4f}\t Correct pred:{cnt}, Total:{total}, Average Hamming Distance:{total_hd/total}", end="\r")

  curr_log += f"loss:{np.mean(losses):.4f}\t"
  print_and_save(curr_log, f"{log_dir}/qram_log.txt")

  if (epoch+1) % qram_save_step == 0:
    torch.save(model.state_dict(), f"{models_dir}/qrams-{address_lines}-{epoch+1}.pth")

# -------------------------
#  QRAM Training End
# -------------------------

