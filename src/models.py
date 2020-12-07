# models.py

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule


class MLP(LightningModule):
    '''Multilayer perceptron.'''
    def __init__(self, C:int, W:int, H:int,
                 num_classes:int, num_layer_1:int=128, num_layer_2:int=256):
        super().__init__()
        self.layer_1 = torch.nn.Linear(C * W * H, num_layer_1)
        self.layer_2 = torch.nn.Linear(num_layer_1, num_layer_2)
        self.layer_3 = torch.nn.Linear(num_layer_2, num_classes)

    def forward(self, x):
        B, C, W, H = x.size()
        x = x.view(B, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

def conv(i:int, o:int, k:int=3, maxpool:bool=True):
    '''i: input channels, o: output channels, k: kernel size, maxpool: use maxpool'''
    blk = OrderedDict([
        ('conv2d', nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, padding=k//2)),
        ('relu', nn.ReLU(inplace=True)),
    ])
    if maxpool: blk['maxpool'] = nn.MaxPool2d(kernel_size=2, stride=2)
    return nn.Sequential(blk)

def twoconv(i:int, o:int, k:int=3):
    return nn.Sequential(OrderedDict([
        ('conv1', conv(i, o, k, maxpool=False)),
        ('conv2', conv(o, o, k)),
    ]))

def fc(i:int, o:int):
    return nn.Sequential(OrderedDict([
        ('linear', torch.nn.Linear(in_features=i, out_features=o)),
        ('relu', nn.ReLU(inplace=True)),
        ('dropout', nn.Dropout(p=0.5)),
    ]))


class CNN(LightningModule):
    '''Simple CNN.'''
    def __init__(self, C, num_classes):
        super().__init__()
        d = 4
        self.features = nn.Sequential(OrderedDict([
            ('conv1', conv(i=C, o=d, k=3)),
            ('conv2', conv(i=d, o=2*d, k=3)),
            ('twoconv1', twoconv(2*d, 4*d, k=3)),
            ('twoconv2', twoconv(4*d, 8*d, k=3)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', fc(i=8 * 8 * (8*d), o=64)),
            ('fc2', fc(i=64, o=128)),
            ('linear', torch.nn.Linear(128, num_classes)),
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class LitModel(LightningModule):
    def __init__(self, datamodule, arch, lr=1e-3):
        super().__init__()
        self.dm = datamodule
        self.arch = arch
        self.lr = lr  # learning rate
        # metrics
        self.accuracy = pl.metrics.Accuracy()
        # save hyper-parameters to self.hparams / log them
        self.save_hyperparameters()

    def forward(self, x):
        return self.arch(x)

    def loss(self, logits, y):
        return F.nll_loss(logits, y)

    def step(self, x, y, prefix=''):
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)
        logger_mode = dict(
            #on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(f'{prefix}_loss', loss, **logger_mode)  # log loss
        self.log(f'{prefix}_acc', acc, **logger_mode)  # log metrics
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(*batch, prefix='train')
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(*batch, prefix='val')

    def test_step(self, batch, batch_idx):
        self.step(*batch, prefix='test')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
        #return SGD(self.parameters(), lr=self.lr, nesterov=True, momentum=0.9)