import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
import numpy as np
import random
from torch.optim import Adam

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_cifar10_dataloaders(batch_size):
    transform = T.Compose(
        [T.ToTensor()])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2, sampler=DistributedSampler(trainset))

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, sampler=DistributedSampler(testset))
    
    return trainloader, testloader

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def accuracy(true, pred, true_one_hot=True, pred_one_hot=True):
    if pred_one_hot:
        pred = pred.argmax(-1)
    if true_one_hot:
        true = true.argmax(-1)
    acc = np.sum((true == pred).astype(np.float32))
    return float(100 * acc / len(true))

def train_and_evaluate(rank, model, optimizer, trainloader, testloader, epochs):
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_running_loss = 0.0
        val_running_loss = 0.0
        train_labels = []
        train_outputs = []
        val_labels = []
        val_outputs = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            train_labels.append(labels)
            train_outputs.append(outputs)

            # print statistics
            train_running_loss += loss

        print(f'[Epoch {epoch + 1}] loss: {train_running_loss / 2000:.3f} accuracy: {accuracy(torch.cat(train_labels, dim=0).detach().cpu().numpy(), torch.cat(train_outputs, dim=0).detach().cpu().numpy(), true_one_hot=False)}')



        net.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                val_labels.append(labels)
                val_outputs.append(outputs)

                # print statistics
                val_running_loss += loss
        print(f'[Epoch {epoch + 1}] val loss: {val_running_loss / 2000:.3f} val accuracy: {accuracy(torch.cat(val_labels, dim=0).detach().cpu().numpy(), torch.cat(val_outputs, dim=0).detach().cpu().numpy(), true_one_hot=False)}')
        
        if rank == 0:
            torch.save(net.module.state_dict(), "ckpt.pt")
    print('Finished Training')    

def main(rank, world_size, epochs):
    ddp_setup(rank, world_size)
    train_dl, test_dl = get_cifar10_dataloaders(batch_size=32)
    model = Net()
    model = DDP(model, device_ids=[rank])
    optimizer = Adam(model.parameters(), lr=0.0005)
    train_and_evaluate(rank, model, optimizer, train_dl, test_dl, epochs)
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(main, args=(world_size, 5), nprocs=world_size)