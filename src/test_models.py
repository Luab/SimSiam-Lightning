from utils import mnist_data, lit_model
from models import CNN, BaseLitModel


def test_CNN():
    cnn = CNN(C=1, num_classes=10)  # Architecture


def test_pl_model():
    mnist = mnist_data()
    cnn = CNN(C=mnist.dims[0], num_classes=mnist.num_classes)  # Architecture
    return BaseLitModel(datamodule=mnist, backbone=cnn, lr=1e-3, flood_height=0)  # Lightning model


# run tests
test_CNN()
test_pl_model()