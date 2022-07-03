import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pennylane as qml
import torch.utils.data as Data
from sklearn.datasets import load_digits
from utils import *
import itertools

name = "model/multi"
log_dir = f"{name}"
models_dir = f"{name}"

device = torch.device("cuda:0")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

batch_size = 16
lr = 1e-3
qaux_wires = 6
qram_wires = 9
n_qaux_layers = n_qram_layers = 3
n_qclassifier_layers = 5
seed = 1234
n_qrams = 1
qram_epochs = 200
qram_start_epoch = 0
qaux_epochs = 100
qcls_epochs = 100
qcls_start_epochs = 0
save_step = 50

data = load_digits().data
target = load_digits().target

indice = np.where((target == 0) | (target == 1))
# Only keep samples with class 0 or 1.
imgs, labels = torch.from_numpy(data[indice]).float(), torch.from_numpy(target[indice]).float()

zero_indices = np.where((labels == 0))
one_indices = np.where((labels == 1))

# Make address dataset corresconding to digits.
adds = []
for i in range(len(imgs)):
    bin = np.binary_repr(i, width=9)
    adds.append([int(char) for char in bin])
adds = torch.FloatTensor(adds)

# Make dataloader for a single qram.
dataset = Data.TensorDataset(adds, imgs, labels)
dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.85*len(imgs)), len(imgs)-int(0.85*len(imgs))])
train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = Data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=False)

dataset = Data.TensorDataset(adds[zero_indices], imgs[zero_indices], labels[zero_indices])
zero_dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.85*len(imgs[zero_indices])), len(imgs[zero_indices])-int(0.85*len(imgs[zero_indices]))])
zero_train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
zero_test_loader = Data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=False)

dataset = Data.TensorDataset(adds[one_indices], imgs[one_indices], labels[one_indices])
one_dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.85*len(imgs[one_indices])), len(imgs[one_indices])-int(0.85*len(imgs[one_indices]))])
one_train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
one_test_loader = Data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=False)

# Define classical and quantum devices.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
qdevice = qml.device("default.qubit", wires=9)

# Define auxilary variational quantum circuit.
@qml.qnode(qdevice, interface='torch')
def qaux(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, normalize=True, wires=range(qaux_wires))
    qml.RandomLayers(weights=weights, wires=range(qaux_wires),seed=seed)
    return qml.probs(wires=range(qaux_wires))

qaux_weights = [{'weights':(n_qaux_layers, 20)} for _ in range(n_qrams)]
qaux_layers = [qml.qnn.TorchLayer(qaux, i, init_method=torch.nn.init.normal_) for i in qaux_weights]

# Define qram variational quantum circuit.
@qml.qnode(qdevice, interface='torch')
def qram(inputs, params, weights):
    qml.AngleEmbedding(features=inputs, wires=range(qram_wires))
    for l in range(n_qram_layers):
        j = 0
        for i in range(qram_wires):
            qml.RY(params[l*36+j],wires=i)
            qml.RY(params[l*36+j+1],wires=(i+1)%qram_wires)
            qml.CNOT(wires=[i,(i+1)%qram_wires])
            qml.CRZ(params[l*36+j+2], wires=[i,(i+1)%qram_wires])
            qml.PauliX(wires=(i+1)%qram_wires)
            qml.CRX(params[l*36+j+3],wires=[i,(i+1)%qram_wires])
            j += 4
    qml.StronglyEntanglingLayers(weights=weights, wires=range(qram_wires))
    return qml.probs(wires=[0,1,2,6,7,8])

qram_weights = [{"params":n_qram_layers*36, "weights":(n_qram_layers, qram_wires, n_qram_layers)} for _ in range(n_qrams)]
qram_layers = [qml.qnn.TorchLayer(qram, i, init_method=torch.nn.init.normal_) for i in qram_weights]

# Define quantum classifier without embedding.
@qml.qnode(qdevice, interface='torch')
def qclassifier(inputs, params, weights, params2):
    # Use same architecture with qram.
    qml.AngleEmbedding(features=inputs, wires=range(qram_wires))
    for l in range(n_qram_layers):
        j = 0
        for i in range(qram_wires):
            qml.RY(params[l*36+j],wires=i)
            qml.RY(params[l*36+j+1],wires=(i+1)%qram_wires)
            qml.CNOT(wires=[i,(i+1)%qram_wires])
            qml.CRZ(params[l*36+j+2], wires=[i,(i+1)%qram_wires])
            qml.PauliX(wires=(i+1)%qram_wires)
            qml.CRX(params[l*36+j+3],wires=[i,(i+1)%qram_wires])
            j += 4
    qml.StronglyEntanglingLayers(weights=weights, wires=range(qram_wires))

    # Define quantum classifier.
    for l in range(n_qclassifier_layers):
        j = 0
        for i in range(qram_wires):
            qml.RY(params2[l*36+j],wires=i)
            qml.RY(params2[l*36+j+1],wires=(i+1)%qram_wires)
            qml.CNOT(wires=[i,(i+1)%qram_wires])
            qml.CRZ(params2[l*36+j+2], wires=[i,(i+1)%qram_wires])
            qml.PauliX(wires=(i+1)%qram_wires)
            qml.CRX(params2[l*36+j+3],wires=[i,(i+1)%qram_wires])
            j += 4
    return qml.expval(qml.PauliZ(0))

qclassifier_weights = {"params":n_qram_layers*36, "weights":(n_qram_layers, qram_wires, n_qram_layers), "params2":n_qclassifier_layers*36}
qclassifier_layer = qml.qnn.TorchLayer(qclassifier, qclassifier_weights, init_method=torch.nn.init.normal_)

# Define joint embedding qram model.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qaux_layers = [qaux_layers[i] for i in range(n_qrams)]
        self.qram_layers = [qram_layers[i] for i in range(n_qrams)]
        self.qclassifier_layer = qclassifier_layer

    def qram(self, imgs, adds, cls=0):
        target = self.qaux_layers[cls](imgs)
        pred = self.qram_layers[cls](adds)
        return pred, target

    def forward(self, adds):
        out = self.qclassifier_layer(adds)
        return out

# Make model and optimizer.
model = Model().to(device)
loss_function = torch.nn.MSELoss()
opt_qcla = torch.optim.Adam(model.parameters(), lr=lr)
opt_qrams = [torch.optim.Adam(itertools.chain(model.qaux_layers[i].parameters(), model.qram_layers[i].parameters()), lr=lr) for i in range(n_qrams)]

# -------------------------
#  Phase 1: training qrams
# -------------------------
print("Start training qrams...")

if n_qrams == 1:
    dataloader = [dataloader]
elif n_qrams == 2:
    dataloader = [zero_dataloader, one_dataloader]

for epoch in range(qram_start_epoch, qram_epochs):
    curr_log = f"epoch {epoch+1}/{qram_epochs}\t"

    for cls in range(n_qrams):
        losses = []
        for batch, (adds, imgs, labels) in enumerate(dataloader[cls]):

            opt_qrams[cls].zero_grad()
            pred, target = model.qram(imgs, adds, cls)
            loss = torch.mean(torch.sum((pred - target)**2, 1))
            loss.backward()
            opt_qrams[cls].step()

            losses.append(loss.item())
            print(f"{epoch+1}/{batch}\t cls:{cls}\t loss:{loss.item():.4f}", end="\r")

        curr_log += f"cls{cls}-loss:{np.mean(losses):.4f}\t"
    print_and_save(curr_log, f"{log_dir}/qram_log.txt")

    if (epoch+1) % save_step == 0:
        torch.save({'qaux_0_state_dict': model.qaux_layers[0].state_dict(),
                    'qaux_1_state_dict': model.qaux_layers[1].state_dict(),
                    'qram_0_state_dict': model.qram_layers[0].state_dict(),
                    'qram_1_state_dict': model.qram_layers[1].state_dict()},
                    f"{models_dir}/qrams-{epoch+1}.pth")

    # Fix qaux layer parameters.
    if (epoch+1) == qaux_epochs:
        for layer in model.qaux_layers:
            for param in layer.parameters():
                param.requires_grad = False


# -------------------------------
#  Phase 2: training qclassifier
# -------------------------------
print("Start training qclassifier...")

if n_qrams == 1:
    train_loader = [train_loader]
    test_loader = [test_loader]
elif n_qrams == 2:
    train_loader = [zero_train_loader, one_train_loader]
    test_loader = [zero_test_loader, one_test_loader]

for epoch in range(qcls_start_epochs, qcls_epochs):
    curr_log = f"epoch {epoch+1}/{qcls_epochs}\t"
    train_losses = test_losses = []
    train_acc = test_acc = 0

    for cls in range(n_qrams):
        # Copy trained qram parameters.
        for p_qram, p_qclassifier in zip(model.qram_layers[cls].parameters(), list(model.parameters())[:-1]):
            p_qclassifier.data.copy_(p_qram.data)
            p_qclassifier.requires_grad = False

        for batch, (adds, imgs, labels) in enumerate(train_loader[cls]):
            opt_qcla.zero_grad()
            out = nn.Sigmoid()(model(adds))
            loss = nn.BCELoss()(out, labels)
            loss.backward()
            opt_qcla.step()

            dist = torch.abs(labels - out)
            train_acc += len(dist[dist<.5])
            train_losses.append(loss.item())
            print(f"{epoch+1}:{batch}\t cls:{cls}\t train_loss:{loss.item():.4f}", end="\r")

        for batch, (adds, imgs, labels) in enumerate(test_loader[cls]):
            # Test the model.
            out = nn.Sigmoid()(model(adds))
            loss = nn.BCELoss()(out, labels)

            dist = torch.abs(labels - out)
            test_acc += len(dist[dist<.5])
            test_losses.append(loss.item())
            print(f"{epoch+1}:{batch}\t cls:{cls}\t test_loss:{loss.item():.4f}", end="\r")

    curr_log += f"train loss:{np.mean(train_losses):.4f}\t test loss:{np.mean(test_losses):.4f}\t"
    curr_log += f"train acc:{train_acc/(len(zero_train_loader.dataset)+len(one_train_loader.dataset)):.4f}\t test acc:{test_acc/(len(zero_test_loader.dataset)+len(one_test_loader.dataset)):.4f}"
    print_and_save(curr_log, f"{log_dir}/model-log.txt")

    if (epoch+1) % save_step == 0:
        torch.save({'model_state_dict': model.state_dict()}, f"{models_dir}/model-{epoch+1}.pth")
