import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

import torch
import torchvision.transforms as T

import wandb
wandb.login()

## internal
from src.callbacks import ImagePredictionLogger
from src.dataset import ComposeMany, MNISTDataModule2, ComposeManyTorch
from src.losses import log_softmax, simsiam_loss
from src.models import accuracy, feature_std, CNN, BaseLitModel, SimSiam


def cli_main():
    pl.seed_everything(1234)

    ## Augmentator
    AUG_KWARGS = dict(border_mode=A.cv2.BORDER_CONSTANT, value=0,
                      interpolation=A.cv2.INTER_LANCZOS4)

#     transforms = ComposeMany([
#         #A.ElasticTransform(p=0.5, sigma=1, alpha=3, alpha_affine=0, **AUG_KWARGS),
#         #A.ElasticTransform(p=0.5, sigma=1, alpha=1, alpha_affine=3, **AUG_KWARGS),
#         A.ShiftScaleRotate(p=1.0, scale_limit=.2, rotate_limit=0, **AUG_KWARGS),
#         A.ShiftScaleRotate(p=1.0, scale_limit=0, rotate_limit=25, **AUG_KWARGS),
#         A.Normalize(mean=(0.0,), std=(1,)),  # , max_pixel_value=255),
#         ToTensorV2()
#     ], n_aug=2)

    transforms = ComposeManyTorch([
        T.RandomResizedCrop(28, scale=(0.6, 1.0)),
        #T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        #T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=28//20*2+1, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize([0.1307], [0.3081]),
    ], n_aug=2)

    ## Lightning datamodule. Comes with its own train / val / test dataloader.
    mnist = MNISTDataModule2(data_dir='../data/',
                             batch_size=128,
                             train_transforms=transforms,
                             val_transforms=transforms)
    mnist.prepare_data()
    mnist.setup()

    ## Backbone architecture
    cnn = CNN(num_channels=mnist.dims[0], num_classes=mnist.num_classes,
              maxpool=False, wpool=5, p_dropout=0.0)
    simsiam = SimSiam(backbone=cnn, p_dropout=0.0)

    model = BaseLitModel(
        datamodule=mnist,
        backbone=simsiam, loss_func=simsiam_loss, metrics=(('featstd', feature_std),),
        lr=0.05 * mnist.batch_size / 256,
        #backbone=cnn, loss_func=log_softmax, metrics=(('acc', accuracy),),
        #lr=1e-4,
        #flood_height=0.03
    )

    proj = 'SimSiam-Lightning'
    wandb_logger = WandbLogger(project=proj, job_type='train')
    callbacks = [
        LearningRateMonitor(),  # log the LR
        #ImagePredictionLogger(mnist.val_dataloader(batch_size=32), n_samples=32),
    ]

    trainer = Trainer(
        max_epochs=100, gpus=-1,  # all GPUs
        logger=wandb_logger, callbacks=callbacks,
        accumulate_grad_batches=1, gradient_clip_val=0,  # 0.5
        progress_bar_refresh_rate=20,
        #fast_dev_run=True,
    )
    trainer.fit(model, mnist)


if __name__ == '__main__':
    cli_main()