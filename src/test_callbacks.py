from callbacks import ImagePredictionLogger
from utils import mnist_data, lit_model


def test_ImagePredictionLogger():
    mnist = mnist_data()  # datamodule
    return ImagePredictionLogger(mnist.val_dataloader())


def test_get_pred_probs():
    # this also tests the constructor
    img_logger = test_ImagePredictionLogger()
    img_logger.get_pred_probs(lit_model())


# run tests
test_ImagePredictionLogger()
test_get_pred_probs()