# models.py

## regular
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam  # , SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from src.losses import flood


## constants
LOG_KWARGS = dict(
    #on_step=True, on_epoch=True, prog_bar=True, logger=True
)


# TODO: tests
## Module functions

def conv(i : int, o : int, k : int=3,
         batchnorm : bool = True, relu=nn.LeakyReLU, maxpool : bool = True):
    '''Conv (+ BatchNorm) + ReLU (+ MaxPool)
    i: input channels, o: output channels, k: kernel size, relu: relu type
    '''
    convblock = OrderedDict([
        ('conv2d', nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, padding=k//2)),
    ])
    if batchnorm: convblock['bn'] = nn.BatchNorm2d(num_features=o)
    convblock['relu'] = relu(inplace=True)
    if maxpool: convblock['maxpool'] = nn.MaxPool2d(kernel_size=2, stride=2)
    return nn.Sequential(convblock)


def twoconv(i : int, o : int, k : int = 3, batchnorm : bool = True,  relu=nn.LeakyReLU):
    '''Conv (+ BatchNorm) + ReLU + Conv (+ BatchNorm) + ReLU + MaxPool
    i: input channels, o: output channels, k: kernel size, relu: relu type
    '''
    return nn.Sequential(OrderedDict([
        ('conv1', conv(i, o, k, batchnorm=batchnorm, relu=relu, maxpool=False)),
        ('conv2', conv(o, o, k, batchnorm=batchnorm, relu=relu, maxpool=True)),
    ]))


def fc(i : int, o : int, relu=nn.LeakyReLU):
    '''Linear + ReLU + Dropout.
    i: input channels, o: output channels, relu: relu class
    '''
    return nn.Sequential(OrderedDict([
        ('linear', torch.nn.Linear(in_features=i, out_features=o)),
        ('relu', relu(inplace=True)),
        ('dropout', nn.Dropout(p=0.5)),
    ]))


## Metrics

def accuracy(batch, forward_callable, device):
    x, y = batch
    logits = forward_callable(x)
    log_y_hat = F.log_softmax(logits, dim=1)  # log probability
    func = pl.metrics.Accuracy().to(device)
    return func(log_y_hat, y)


def feature_std(batch, forward_callable, device):
    x, _ = batch
    z1, z2, p1, p2 = forward_callable(x)
    #import ipdb; ipdb.set_trace()
    z1 = F.normalize(z1, dim=1)
    return z1.std(dim=1).mean(dim=0)  # B, d


## Architectures

class MLP(torch.nn.Module):
    '''Multilayer perceptron.'''
    def __init__(self, C : int, W : int, H : int,
                 num_classes : int, num_layer_1 : int = 128, num_layer_2 : int = 256):
        super().__init__()
        self.layer_1 = torch.nn.Linear(C * W * H, num_layer_1)
        self.layer_2 = torch.nn.Linear(num_layer_1, num_layer_2)
        self.layer_3 = torch.nn.Linear(num_layer_2, num_classes)
        relu = F.leaky_relu

    def forward(self, x):
        B, C, W, H = x.size()
        x = x.view(B, -1)
        x = self.layer_1(x)
        x = relu(x)
        x = self.layer_2(x)
        x = relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNN(torch.nn.Module):
    '''CNN based on VGG.'''
    def __init__(self, num_channels : int, num_classes : int):
        super().__init__()
        d = 16  # 4
        wpool = 16  # 8
        fc_o = 512  # 128
        self.features = nn.Sequential(
            OrderedDict([('conv1', conv(i=num_channels, o=d, k=3)),
                         ('conv2', conv(i=d, o=2*d, k=3)),
                         ('twoconv1', twoconv(2*d, 4*d, k=3)),
                         ('twoconv2', twoconv(4*d, 8*d, k=3)),
                        ]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(wpool, wpool))
        self.classifier = nn.Sequential(  # two fc layers that output the logits
            OrderedDict([('fc1', fc(i=wpool * wpool * (8*d), o=fc_o//2)),
                         ('fc2', fc(i=fc_o//2, o=fc_o)),
                         ('linear', torch.nn.Linear(fc_o, num_classes)),
                        ]))

    def forward(self, x):
        x = self.features(x[:, [0]])  # with only the first channel (in case of many augs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # logits


class SimSiam(torch.nn.Module):
    '''Simple Siamese-Net.
    backbone: torch module with a `features` property.
    aug: random augmentation function.
    '''
    def __init__(self, backbone : torch.nn.Module):  #, aug):
        super().__init__()
        d, wpool, fc_o = 16, 8, 128  # 4, 16, 512
        #self.aug = aug
        self.f = nn.Sequential(
            OrderedDict([('features', backbone.features),
                         ('avgpoool', nn.AdaptiveAvgPool2d(output_size=(wpool, wpool))),
                         ('flatten', nn.Flatten(start_dim=1)),
                         ('fc', fc(i = wpool * wpool * (8*d), o = fc_o)),
                        ]))
        self.h = nn.Sequential(
            OrderedDict([('fc1', fc(i=fc_o, o=fc_o//2)),  # bottleneck
                         ('fc2', fc(i=fc_o//2, o=fc_o)),]))

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        x1, x2 = x[:, [0]], x[:, [1]]  # two random augmentations
        z1, z2 = self.f(x1), self.f(x2)  # projections
        p1, p2 = self.h(z1), self.h(z2)  # centroid predictions
        return z1, z2, p1, p2

#     def D(self, p, z):
#         '''Negative cosine similarity.'''
#         p = F.normalize(p, dim=1)  # l2-normalize
#         z = F.normalize(z, dim=1)
#         return -(p*z).sum(dim=1).mean()

#     def loss(self, z1, z2, p1, p2):
#         return D(p1, stopgrad(z2))/2 + D(p2, stopgrad(z1))/2


## Lightning modules

class BaseLitModel(LightningModule):
    '''A module for all those nice PyTorch-Lightning features.'''
    def __init__(self, datamodule=None, backbone=None, loss_func=None, metrics : tuple = (),
                 lr : float = 1e-3, batch_size : int = 32, flood_height: float = 0):
        super().__init__()
        self.dm = datamodule
        self.backbone = backbone  # model architecture
        self.lr = lr  # learning rate
        self.batch_size = batch_size
        self.flood_height = flood_height  # 0.03 # flood the loss
        self.loss_func = loss_func
        self.metrics = metrics
        self.metric_names, self.metric_funcs = zip(*metrics) if metrics else ((), ())
        print(f'Logging metrics: {list(self.metric_names)}')
        self.save_hyperparameters()  # save hyper-parameters to self.hparams, and log them

    def forward(self, x):
        return self.backbone(x)

    def loss(self, batch, flood_height : float = 0):
        loss = self.loss_func(batch, self.forward)
        if flood_height > 0: loss = flood(loss, flood_height)  # with flooding
        return loss

    def step(self, batch, prefix : str = '', flood_height : float = 0):
        # Log loss
        loss = self.loss(batch, flood_height)
        self.log(f'{prefix}_loss', loss, **LOG_KWARGS)
        # Log metrics
        metric_results = [metric_func(batch, forward_callable=self.forward, device=self.device)
                          for metric_func in self.metric_funcs]
        for name, metric in zip(self.metric_names, metric_results):
            self.log(f'{prefix}_{name}', metric, **LOG_KWARGS)
        return loss

    def training_step(self, batch, batch_idx):
        lr = self.optimizer.param_groups[0]['lr']
        loss = self.step(batch, prefix='train', flood_height=self.flood_height)
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, prefix='val', flood_height=0)

    def test_step(self, batch, batch_idx):
        self.step(batch, prefix='test', flood_height=0)

    def configure_optimizers(self):
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        #self.optimizer = SGD(self.parameters(), lr=self.lr, nesterov=True, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=10, cooldown=0)
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.scheduler,
                'monitor': 'val_loss',}