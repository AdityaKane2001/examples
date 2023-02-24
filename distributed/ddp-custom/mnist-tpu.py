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

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from torchvision import datasets, transforms


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5554"
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_mnist_datasets():
    transform = T.Compose(
        [T.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    return trainset, testset


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

def train_and_evaluate(epochs):
    criterion = nn.CrossEntropyLoss()

    train_dataset, test_dataset = SERIAL_EXEC.run(get_mnist_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=2,
        drop_last=True)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        drop_last=True)
    
    lr = 0.0001 * xm.xrt_world_size()

    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    

    def train_epoch(trainloader):
        train_running_loss = 0.0
        
        train_labels = []
        train_outputs = []
        

        model.train()

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer)

            train_labels.append(labels)
            train_outputs.append(outputs)

            # print statistics
            train_running_loss += loss
        return train_labels, train_outputs, train_running_loss

    


    def eval_epoch():
        val_labels = []
        val_outputs = []
        val_running_loss = 0.0

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs])
                loss = criterion(outputs, labels)

                val_labels.append(labels)
                val_outputs.append(outputs)

                # print statistics
                val_running_loss += loss
                return val_labels, val_outputs, val_running_loss

    for epoch in range(epochs):
        para_loader = pl.ParallelLoader(trainloader, [device])
        train_labels, train_outputs, train_running_loss = train_epoch(para_loader.per_device_loader(device))

        para_loader = pl.ParallelLoader(testloader, [device])
        val_labels, val_outputs, val_running_loss = eval_epoch(para_loader.per_device_loader(device))

        xm.master_print(f'[Epoch {epoch + 1}] train loss: {train_running_loss / 2000:.3f} accuracy: {accuracy(torch.cat(train_labels, dim=0).detach().cpu().numpy(), torch.cat(train_outputs, dim=0).detach().cpu().numpy(), true_one_hot=False)}')
        xm.master_print(f'[Epoch {epoch + 1}] val loss: {val_running_loss / 2000:.3f} val accuracy: {accuracy(torch.cat(val_labels, dim=0).detach().cpu().numpy(), torch.cat(val_outputs, dim=0).detach().cpu().numpy(), true_one_hot=False)}')
    

 
    print('Finished Training')    


SERIAL_EXEC = xmp.MpSerialExecutor()
WRAPPED_MODEL = xmp.MpModelWrapper(Net())

def _mp_fn(rank, epochs):

  torch.set_default_tensor_type('torch.FloatTensor')
  train_and_evaluate(epochs)
  
  
  
xmp.spawn(_mp_fn, args=(10,), nprocs=xm.xrt_world_size(),
          start_method='fork')