import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
import numpy as np
import random
from torch.optim import Adam

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl

def accuracy(true, pred, true_one_hot=True, pred_one_hot=True):
    if pred_one_hot:
        pred = pred.argmax(-1)
    if true_one_hot:
        true = true.argmax(-1)
    acc = np.sum((true == pred).astype(np.float32))
    return float(100 * acc / len(true))

class Net(pl.LightningModule):
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

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.001)

def get_cifar10_dataloaders(batch_size):
    transform = T.Compose(
        [T.ToTensor()])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader

mnist_train, mnist_val = get_cifar10_dataloaders(1024)

image_classifier = Net()

# most basic trainer, uses good defaults (8 TPU Cores)
# training only for 20 epochs for demo purposes. For training longer simply adjust number below.
trainer = pl.Trainer(tpu_cores=8, max_epochs=2)    
trainer.fit(image_classifier, mnist_train, mnist_val)

trainer.test(image_classifier, ckpt_path=None, dataloaders=mnist_val)