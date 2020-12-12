# models.py

# regular
from collections import OrderedDict

# PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Lightning
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule


LOG_KWS = dict(
    #on_step=True, on_epoch=True, prog_bar=True, logger=True
)


def conv(i : int, o : int, k : int=3,
         batchnorm : bool = True, relu=nn.LeakyReLU, maxpool : bool = True):
    '''Conv (+ BatchNorm) + ReLU (+ MaxPool)
    i: input channels, o: output channels, k: kernel size, relu: relu type
    '''
    convblock = OrderedDict([
        ('conv2d', nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, padding=k//2)),
        #('bn', nn.BatchNorm2d(num_features=o)),
        #('relu', relu(inplace=True))
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
    i: input channels, o: output channels, relu: relu type
    '''
    return nn.Sequential(OrderedDict([
        ('linear', torch.nn.Linear(in_features=i, out_features=o)),
        ('relu', relu(inplace=True)),
        ('dropout', nn.Dropout(p=0.5)),
    ]))


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


# class CNN(LightningModule):
class CNN(torch.nn.Module):
    '''CNN based on VGG.'''
    def __init__(self, C : int, num_classes : int):
        super().__init__()
        d = 4  # 16
        self.features = nn.Sequential(
            OrderedDict([('conv1', conv(i=C, o=d, k=3)),
                         ('conv2', conv(i=d, o=2*d, k=3)),
                         ('twoconv1', twoconv(2*d, 4*d, k=3)),
                         ('twoconv2', twoconv(4*d, 8*d, k=3)),])
        )
        wpool = 8
        fc_o = 128
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(wpool, wpool))
        # two fc layers that output the logits
        self.classifier = nn.Sequential(
            OrderedDict([('fc1', fc(i=wpool * wpool * (wpool*d), o=fc_o//2)),
                         ('fc2', fc(i=fc_o//2, o=fc_o)),
                         ('linear', torch.nn.Linear(fc_o, num_classes)),])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # logits


class LitModel(LightningModule):
    '''Module that hosts all the nice PyTorch-Lightning functions.'''
    def __init__(self, datamodule, backbone, lr : float = 1e-3, batch_size : int = 32,
                 flood : bool = False):
        super().__init__()
        self.dm = datamodule
        self.backbone = backbone  # model architecture
        self.lr = lr  # learning rate
        self.batch_size = batch_size
        self.flood = flood  # use flooding
        self.save_hyperparameters()  # save hyper-parameters to self.hparams, and log them
        self.accuracy = pl.metrics.Accuracy()  # metrics

    def forward(self, x):
        return self.backbone(x)

    def loss(self, y_hat, y, flood : float = 0.05):
        '''Loss flooding (see arXiv).'''
        loss = F.nll_loss(y_hat, y)  # yhat is a log prob
        if self.flood: loss = torch.abs(loss - flood) + flood
        return loss

    def step(self, x, y, prefix='', flood : float = 0.05):
        logits = self(x)  # logits
        log_y_hat = F.log_softmax(logits, dim=1)  # log probability
        loss = self.loss(log_y_hat, y, flood)  # w flooding
        acc = self.accuracy(log_y_hat, y)
        self.log(f'{prefix}_loss', loss, **LOG_KWS)  # log metrics
        self.log(f'{prefix}_acc', acc, **LOG_KWS)
        return loss, acc

    def training_step(self, batch, batch_idx):
        # default flooding: 0.001 * 50 = 0.05
        lr = self.optimizer.param_groups[0]['lr']
        loss, acc = self.step(*batch, prefix='train', flood=0.03)#, flood=lr * 50)
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(*batch, prefix='val', flood=0)

    def test_step(self, batch, batch_idx):
        self.step(*batch, prefix='test', flood=0)

    def configure_optimizers(self):
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        #self.optimizer = SGD(self.parameters(), lr=self.lr, nesterov=True, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=10, cooldown=0)
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.scheduler,
                'monitor': 'val_loss',}