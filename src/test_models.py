import torch

from utils import mnist_data, lit_model
from models import CNN, BaseLitModel, SimSiam


def test_CNN():
    cnn = CNN(num_channels=1, num_classes=10, maxpool=True)
    cnn = CNN(num_channels=1, num_classes=10, maxpool=False)
    cnn = CNN(num_channels=1, num_classes=10, maxpool=True, p_dropout=0.0)


def test_SimSiam():
    cnn = CNN(num_channels=1, num_classes=10, maxpool=False, wpool=5, p_dropout=0.0)
    simsiam = SimSiam(backbone=cnn, p_dropout=0.0)
    return simsiam


def test_SimSiam_forward():
    simsiam = test_SimSiam()
    ## BatchNorm needs more than 1 sample to estimate the std.
    x = torch.rand(32, 2, 28, 28)
    simsiam(x)


def test_pl_model():
    mnist = mnist_data()
    cnn = CNN(num_channels=mnist.dims[0], num_classes=mnist.num_classes)  # Architecture
    BaseLitModel(datamodule=mnist, backbone=cnn, lr=1e-3, flood_height=0)  # Lightning model
    cnn = CNN(num_channels=mnist.dims[0], num_classes=mnist.num_classes, maxpool=False)
    BaseLitModel(datamodule=mnist, backbone=cnn, lr=1e-3, flood_height=0)


## Run tests
test_CNN()
test_pl_model()