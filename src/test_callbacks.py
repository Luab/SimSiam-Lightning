import albumentations as A
from albumentations.pytorch import ToTensorV2

from callbacks import ImagePredictionLogger, knnMonitorLogger
from utils import mnist_data, lit_model
from dataset import ComposeMany


def test_ImagePredictionLogger():
    mnist = mnist_data()  # datamodule
    return ImagePredictionLogger(mnist.val_dataloader())


def test_knnMonitorLogger():
    mnist = mnist_data()  # datamodule
    test_transforms = ComposeMany([A.Normalize(mean=(0.0,), std=(1,)), ToTensorV2()])
    memory_dataloader = mnist.train_dataloader(transforms=test_transforms)
    test_dataloader = mnist.test_dataloader(transforms=test_transforms)
    return knnMonitorLogger(memory_dataloader, test_dataloader, knn_k=1, knn_t=1)


def test_get_pred_probs():
    # this also tests the constructor
    img_logger = test_ImagePredictionLogger()
    img_logger.get_pred_probs(lit_model())


# run tests
test_ImagePredictionLogger()
test_get_pred_probs()