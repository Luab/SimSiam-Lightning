# models.py

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from src.losses import flood


## Constants
LOG_KWARGS = dict(
    #on_step=True, on_epoch=True, prog_bar=True, logger=True
)


# TODO: tests
## Module functions

def conv(i : int, o : int, k : int = 3,
         bn : bool = True, relu=nn.LeakyReLU, maxpool : bool = False):
    '''
    Conv2d (+ BatchNorm2d) (+ ReLU) (+ MaxPool2d).

    Args:
        i: input channels
        o: output channels
        k: kernel size
        bn: apply batch normalization.
        relu: relu type. If False or None, nn.Identity will be used instead.
        maxpool: apply maxpooling (2x2) at the end.
    '''

    convblock = OrderedDict([
        ('conv2d', nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, padding=k//2)),
        ('bn', nn.BatchNorm2d(num_features=o) if bn else nn.Identity()),
        ('relu', relu(inplace=True) if relu else nn.Identity()),
        ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else nn.Identity()),
    ])
    return nn.Sequential(convblock)


def twoconv(i : int, o : int, k : int = 3,
            bn : bool = True, relu=nn.LeakyReLU, maxpool : bool = False):
    '''
    Conv2d (+ BatchNorm2d) (+ ReLU) +
    Conv2d (+ BatchNorm2d) (+ ReLU) (+ MaxPool2d).

    Args:
        i: input channels
        o: output channels
        k: kernel size
        bn: apply batch normalization.
        relu: relu type. If False or None, nn.Identity will be used.
        maxpool: apply maxpooling (2x2) at the end.
    '''

    relu = relu or nn.Identity
    return nn.Sequential(OrderedDict([
        ('conv1', conv(i, o, k, bn, relu, maxpool=False)),
        ('conv2', conv(o, o, k, bn, relu, maxpool=maxpool)),
    ]))


def fc(i : int, o : int, bn : bool = True, relu=nn.LeakyReLU, p_dropout=0.5):
    '''
    Linear (+ BatchNorm1d) (+ ReLU) (+ Dropout).

    Args:
        i: input channels
        o: output channels
        bn: apply batch normalization.
        relu: relu type. If False or None, nn.Identity will be used instead.
        p_dropout: probability of applying dropout per element.
    '''

    return nn.Sequential(OrderedDict([
        ('linear', torch.nn.Linear(in_features=i, out_features=o)),
        ('bn', nn.BatchNorm1d(num_features=o) if bn else nn.Identity()),
        ('relu', relu(inplace=True) if relu else nn.Identity()),
        ('dropout', nn.Dropout(p=p_dropout)),
    ]))


## Metrics

def accuracy(batch, forward_callable, device):
    '''To track the accuracy (% classified correctly).'''
    x, y = batch
    logits = forward_callable(x)
    ## Log probability
    log_y_hat = F.log_softmax(logits, dim=1)
    func = pl.metrics.Accuracy().to(device)
    return func(log_y_hat, y)


def feature_std(batch, forward_callable, device):
    '''To track the standard deviation of a feature across samples, averaged across features.'''
    x, _ = batch
    z1, z2, p1, p2 = forward_callable(x)
    z1 = F.normalize(z1, dim=1)
    ## (B, d)
    return z1.std(dim=0).mean()


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
    '''CNN based on the VGG net.'''

    def __init__(self, num_channels : int, num_classes : int,
                 maxpool : bool = True,
                 wpool : int = 5,
                 p_dropout : float = 0.5):
        super().__init__()

        ## Kernel size.
        k = 3
        ## Channel multiplier. output channels = d x input channels.
        d = 2
        ## Average-pooling output size.
        self.maxpool = maxpool
        wpool = 1 if maxpool else wpool # 16
        self.wpool = wpool
        ## Input features at the fc layer.
        fc_d = (8 * d) * wpool * wpool  # 128  # 512

        ## If maxpool is True, there will be H x W // 2**4 features.
        self.features = nn.Sequential(
            OrderedDict([('conv1', conv(i=num_channels, o=d, k=k, maxpool=maxpool)),  # 14x14
                         ('conv2', conv(i=d, o=2*d, k=k, maxpool=maxpool)),  # 7x7
                         ('twoconv1', twoconv(i=2*d, o=4*d, k=k, maxpool=maxpool)),  # 3x3
                         ('twoconv2', twoconv(i=4*d, o=8*d, k=k, maxpool=maxpool)),  # 1x1
                        ]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(wpool, wpool))
        ## A bottleneck of three fc layers that output the logits.
        self.classifier = nn.Sequential(
            OrderedDict([('fc1', fc(i=fc_d, o=fc_d//8, p_dropout=p_dropout)),
                         ('fc2', fc(i=fc_d//8, o=fc_d, p_dropout=p_dropout)),
                         ('linear', torch.nn.Linear(fc_d, num_classes))
                        ]))

    def forward(self, x):
        ## With only the first channel (in case of many augs).
        x = self.features(x[:, [0]])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        ## Logits
        return self.classifier(x)


class SimSiam(torch.nn.Module):
    '''
    Simple Siamese Net.

    Args:
        backbone: torch module with a `features` property.
        aug: random augmentation function.
    '''

    def __init__(self,
                 backbone : torch.nn.Module,
                 projection_d : int = 2048,
                 prediction_d : int = 2048,
                 p_dropout : float = 0.0):
        super().__init__()

        ## Output channels = d x input channels.
        d = 2
        ## Average pooling output size.
        wpool = backbone.wpool
        ## Input, hidden, output features of the projection MLP.
        fc_i = (8 * d) * wpool * wpool  # 4, 16, 512
        fc_d = projection_d
        fc_o = prediction_d

        self.backbone = backbone

        ## Three fc layers that output the projection.
        self.projection_mlp = nn.Sequential(
            OrderedDict([('fc1', fc(i=fc_i, o=fc_d, bn=True, p_dropout=p_dropout)),
                         ('fc2', fc(i=fc_d, o=fc_d, bn=True, p_dropout=p_dropout)),
                         ('fc3', fc(i=fc_d, o=fc_o, bn=True, relu=None, p_dropout=p_dropout)),
                        ]))

        ## A bottleneck of two fc layers that output the prediction.
        self.prediction_mlp = nn.Sequential(
            OrderedDict([('fc1', fc(i=fc_o, o=fc_o//4, bn=True, p_dropout=p_dropout)),
                         ('fc2', fc(i=fc_o//4, o=fc_o, bn=False, relu=None, p_dropout=p_dropout)),
                        ]))

        self.f = nn.Sequential(
            OrderedDict([('features', self.backbone.features),
                         ('avgpool', nn.AdaptiveAvgPool2d(output_size=(wpool, wpool))),
                         ('flatten', nn.Flatten(start_dim=1)),
                         ('projection_mlp', self.projection_mlp),
                        ]))

        self.h = nn.Sequential(
            OrderedDict([
                ('prediction_mlp', self.prediction_mlp),
            ]))

    def forward(self, x):
        x1, x2 = x[:, [0]], x[:, [1]]  ## Two random augmentations
        z1, z2 = self.f(x1), self.f(x2)  ## Projections
        p1, p2 = self.h(z1), self.h(z2)  ## Centroid predictions
        return z1, z2, p1, p2


## Lightning modules

class BaseLitModel(LightningModule):
    '''A module for all those nice PyTorch-Lightning features.'''

    def __init__(self, datamodule=None, backbone=None, loss_func=None, metrics : tuple = (),
                 lr : float = 1e-3,
                 flood_height: float = 0,
                 optimizer: str = 'adam',
                 *args, **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.dm = datamodule
        ## Backbone architecture
        self.backbone = backbone
        ## Learning rate
        self.lr = lr
        ## Flood the loss
        self.flood_height = flood_height  # 0.03
        self.optimizer_name = optimizer
        self.loss_func = loss_func
        self.metrics = metrics
        self.metric_names, self.metric_funcs = zip(*metrics) if metrics else ((), ())
        self.example_input_array = self.dm.train_dataloader().dataset.dataset[0][0][None]
        print(f'Logging metrics: {list(self.metric_names)}')
        ## Save hyper-parameters to self.hparams, and log them.
        self.save_hyperparameters()

    def forward(self, x):
        return self.backbone(x)

    def loss(self, batch, flood_height : float = 0):
        loss = self.loss_func(batch, self.forward)
        ## With flooding
        if flood_height > 0: loss = flood(loss, flood_height)
        return loss

    def step(self, batch, prefix : str = '', flood_height : float = 0):
        ## Log loss
        loss = self.loss(batch, flood_height)
        self.log(f'{prefix}_loss', loss, **LOG_KWARGS)
        ## Log metrics
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
        if self.optimizer_name == 'adam':
            self.optimizer = Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            self.optimizer = SGD(self.parameters(), lr=self.lr, nesterov=True, momentum=0.9)
        else: raise NotImplemented
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=10, cooldown=0)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss'
        }