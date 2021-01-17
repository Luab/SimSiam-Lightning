from utils import mnist_data, lit_model
from models import CNN, BaseLitModel


def test_CNN():
    cnn = CNN(num_channels=1, num_classes=10, maxpool=True)
    cnn = CNN(num_channels=1, num_classes=10, maxpool=False)
    cnn = CNN(num_channels=1, num_classes=10, maxpool=True, p_dropout=0.0)


def test_pl_model():
    mnist = mnist_data()
    cnn = CNN(num_channels=mnist.dims[0], num_classes=mnist.num_classes)  # Architecture
    BaseLitModel(datamodule=mnist, backbone=cnn, lr=1e-3, flood_height=0)  # Lightning model
    cnn = CNN(num_channels=mnist.dims[0], num_classes=mnist.num_classes, maxpool=False)
    BaseLitModel(datamodule=mnist, backbone=cnn, lr=1e-3, flood_height=0)


## Run tests
test_CNN()
test_pl_model()