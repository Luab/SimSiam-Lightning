from warnings import warn

import albumentations as A

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST


def alb_to_torch_aug(aug):
    '''albumentations.Compose wrapper for PyTorch compatibility.'''
    if type(aug) in [A.Compose or ComposeMany]:
        def wrapped_aug(x):  # , *args, **kwargs):
            #return aug(image=A.np.array(x), *args, **kwargs)['image']
            return aug(image=A.np.array(x))['image']
        return wrapped_aug
    else: return aug


# class ComposeMany(A.Compose):
#     '''Class derived from A.Compose for sampling many augmentations of one image.'''
#     def __init__(self, transforms, n_aug : int=1):  # , *args, **kwargs):
#         '''n_aug: number of augmentations to sample'''
#         super().__init__(transforms)  # , *args, **kwargs)
#         self.n_aug = n_aug

#     def __call__(self, image):  # , *args, **kwargs):
#         import pdb; pdb.set_trace()
#         print(type(image))
#         return {'image':torch.cat(
#                 [
#                     #super(ComposeMany, self).__call__(image=image, *args, **kwargs)['image']
#                     super(ComposeMany, self).__call__(image=image)['image']
#                     for _ in range(self.n_aug)
#                 ], dim=0)
#         }


class MNISTDataModule(LightningDataModule):

    name = "mnist"

    def __init__(
        self,
        data_dir: str = "./",
        val_split: int = 5000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        train_transforms = None,
        val_transforms = None,
        test_transforms = None,
        *args,
        **kwargs,
    ):
        """
        .. figure:: https://miro.medium.com/max/744/1*AO2rIhzRYzFVQlFLx9DM9A.png
            :width: 400
            :alt: MNIST

        Specs:
            - 10 classes (1 per digit)
            - Each image is (1 x 28 x 28)

        Standard MNIST, train, val, test splits and transforms

        Transforms::

            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor()
            ])

        Example::

            from pl_bolts.datamodules import MNISTDataModule

            dm = MNISTDataModule('.')
            model = LitModel()

            Trainer().fit(model, dm)

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
        """
        super().__init__(*args, **kwargs)

        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.train_transforms = alb_to_torch_aug(train_transforms)
        self.val_transforms = alb_to_torch_aug(val_transforms)
        self.test_transforms = alb_to_torch_aug(test_transforms)

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves MNIST files to data_dir
        """
        MNIST(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        MNIST(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST train set removes a subset to use for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.train_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST val set uses a subset of the training set for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()
        dataset = MNIST(self.data_dir, train=True, download=True, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST test set uses the test split

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True
        )
        return loader

    def _default_transforms(self):
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms