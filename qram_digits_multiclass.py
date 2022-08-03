# Code to run digit classification sims with QRAM (for 3 digits and higher)

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
import math

name = "model/multiclass"
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
n_qrams = int(input("Enter Number of classes to classify:"))
qram_epochs = 200
qram_start_epoch = 0
qaux_epochs = 100
qcls_epochs = 100
qcls_start_epochs = 0
save_step = 50

data = load_digits().data
target = load_digits().target

classes = []
for i in range(n_qrams):
    classes.append(int(input(f"Enter label {i+1}:")))
classes_string = "".join([str(i) for i in classes])
classes = np.array(classes)

indices = []
for i in range(n_qrams):
    indice = np.where((target == classes[i]))
    indices.append(indice)
indice = np.concatenate(indices, axis=None)    
print(indice.shape)
imgs, labels = torch.from_numpy(data[indice]).float(), torch.from_numpy(target[indice]).float()
n_qubits = int(math.log2(smallest_power_2(len(imgs))))
qram_wires = n_qubits

# Make address dataset corresponding to digits.
adds = []
for i in range(len(imgs)):
    bin = np.binary_repr(i, width=n_qubits)
    adds.append([int(char) for char in bin])
adds = torch.FloatTensor(adds)

# Make dataloader for all qrams.
dataloader = []
train_loader = []
test_loader = []
train_dataset_size = 0
test_dataset_size = 0

for i in range(n_qrams):
    class_indices = torch.where(labels == classes[i])
    dataset = Data.TensorDataset(adds[class_indices], imgs[class_indices], labels[class_indices])
    dataloader_i = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.85*len(imgs[class_indices])), len(imgs[class_indices])-int(0.85*len(imgs[class_indices]))])
    train_loader_i = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader_i = Data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader.append(dataloader_i)
    train_loader.append(train_loader_i)
    test_loader.append(test_loader_i)
    train_dataset_size += len(train_loader_i.dataset)
    test_dataset_size += len(test_loader_i.dataset)

# Define classical and quantum devices.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
qdevice = qml.device("default.qubit", wires=n_qubits)

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
            qml.RY(params[l*qram_wires*4+j],wires=i)
            qml.RY(params[l*qram_wires*4+j+1],wires=(i+1)%qram_wires)
            qml.CNOT(wires=[i,(i+1)%qram_wires])
            qml.CRZ(params[l*qram_wires*4+j+2], wires=[i,(i+1)%qram_wires])
            qml.PauliX(wires=(i+1)%qram_wires)
            qml.CRX(params[l*qram_wires*4+j+3],wires=[i,(i+1)%qram_wires])
            j += 4
    qml.StronglyEntanglingLayers(weights=weights, wires=range(qram_wires))
    return qml.probs(wires=[0,1,2,6,7,8])

qram_weights = [{"params":n_qram_layers*qram_wires*4, "weights":(n_qram_layers, qram_wires, n_qram_layers)} for _ in range(n_qrams)]
qram_layers = [qml.qnn.TorchLayer(qram, i, init_method=torch.nn.init.normal_) for i in qram_weights]

# Define quantum classifier without embedding.
@qml.qnode(qdevice, interface='torch')
def qclassifier(inputs, params, weights, params2):
    # Use same architecture with qram.
    qml.AngleEmbedding(features=inputs, wires=range(qram_wires))
    for l in range(n_qram_layers):
        j = 0
        for i in range(qram_wires):
            qml.RY(params[l*qram_wires*4+j],wires=i)
            qml.RY(params[l*qram_wires*4+j+1],wires=(i+1)%qram_wires)
            qml.CNOT(wires=[i,(i+1)%qram_wires])
            qml.CRZ(params[l*qram_wires*4+j+2], wires=[i,(i+1)%qram_wires])
            qml.PauliX(wires=(i+1)%qram_wires)
            qml.CRX(params[l*qram_wires*4+j+3],wires=[i,(i+1)%qram_wires])
            j += 4
    qml.StronglyEntanglingLayers(weights=weights, wires=range(qram_wires))

    # Define quantum classifier.
    for l in range(n_qclassifier_layers):
        j = 0
        for i in range(qram_wires):
            qml.RY(params2[l*qram_wires*4+j],wires=i)
            qml.RY(params2[l*qram_wires*4+j+1],wires=(i+1)%qram_wires)
            qml.CNOT(wires=[i,(i+1)%qram_wires])
            qml.CRZ(params2[l*qram_wires*4+j+2], wires=[i,(i+1)%qram_wires])
            qml.PauliX(wires=(i+1)%qram_wires)
            qml.CRX(params2[l*qram_wires*4+j+3],wires=[i,(i+1)%qram_wires])
            j += 4
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qrams)]

qclassifier_weights = {"params":n_qram_layers*qram_wires*4, "weights":(n_qram_layers, qram_wires, n_qram_layers), "params2":n_qclassifier_layers*qram_wires*4}
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
loss_function = torch.nn.CrossEntropyLoss()
opt_qcla = torch.optim.Adam(model.parameters(), lr=lr)
opt_qrams = [torch.optim.Adam(itertools.chain(model.qaux_layers[i].parameters(), model.qram_layers[i].parameters()), lr=lr) for i in range(n_qrams)]

# -------------------------
#  Phase 1: training qrams
# # -------------------------
print("Start training qrams...")

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
    print_and_save(curr_log, f"{log_dir}/qram_{classes_string}_log.txt")

    if (epoch+1) % save_step == 0:
        dict_aux = {f"qaux_{i}_state_dict":model.qaux_layers[i].state_dict() for i in range(n_qrams)}
        dict_qram = {f"qram_{i}_state_dict":model.qram_layers[i].state_dict() for i in range(n_qrams)}
        save_dict = {**dict_aux, **dict_qram}
        torch.save(save_dict, f"{models_dir}/qrams-{classes_string}-{epoch+1}.pth")

    # Fix qaux layer parameters.
    if (epoch+1) == qaux_epochs:
        for layer in model.qaux_layers:
            for param in layer.parameters():
                param.requires_grad = False

# Import parameters if already trained
# checkpoint = torch.load(f"{models_dir}/qrams-012-1.pth")
# model.qaux_layers[0].load_state_dict(checkpoint['qaux_0_state_dict'])
# model.qram_layers[0].load_state_dict(checkpoint['qram_0_state_dict'])
# model.qaux_layers[1].load_state_dict(checkpoint['qaux_1_state_dict'])
# model.qram_layers[1].load_state_dict(checkpoint['qram_1_state_dict'])
# model.qaux_layers[2].load_state_dict(checkpoint['qaux_2_state_dict'])
# model.qram_layers[2].load_state_dict(checkpoint['qram_2_state_dict'])

# -------------------------------
#  Phase 2: training qclassifier
# -------------------------------
print("Start training qclassifier...")

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
            out = model(adds)
            soft_out = nn.Softmax()(out)
            pred = torch.argmax(soft_out, dim=1)
            loss = loss_function(soft_out, labels.long())
            loss.backward()
            opt_qcla.step()

            dist = torch.abs(labels - pred)
            train_acc += len(dist[dist==0])
            train_losses.append(loss.item())
            print(f"{epoch+1}:{batch}\t cls:{cls}\t train_loss:{loss.item():.4f}", end="\r")

        for batch, (adds, imgs, labels) in enumerate(test_loader[cls]):
            # Test the model.
            out = model(adds)
            soft_out = nn.Softmax()(out)
            pred = torch.argmax(soft_out, dim=1)
            loss = loss_function(soft_out, labels.long())

            dist = torch.abs(labels - pred)
            test_acc += len(dist[dist==0])
            test_losses.append(loss.item())
            print(f"{epoch+1}:{batch}\t cls:{cls}\t test_loss:{loss.item():.4f}", end="\r")

    curr_log += f"train loss:{np.mean(train_losses):.4f}\t test loss:{np.mean(test_losses):.4f}\t"
    curr_log += f"train acc:{train_acc/train_dataset_size:.4f}\t test acc:{test_acc/test_dataset_size:.4f}"
    print_and_save(curr_log, f"{log_dir}/model-{classes_string}-log.txt")

    if (epoch+1) % save_step == 0:
        torch.save({'model_state_dict': model.state_dict()}, f"{models_dir}/model-{classes_string}-{epoch+1}.pth")