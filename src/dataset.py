import albumentations as A
import torch
import torchvision.transforms as T
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule  #, LightningDataModule



class ComposeMany(A.Compose):
    '''
    A.Compose derivative for sampling many augmentations of one image.
    Also, provides a PyTorch-like interface to transforms.
    '''

    def __init__(self, transforms, n_aug : int=1, *args, **kwargs):
        '''n_aug: number of augmentations to sample'''
        super().__init__(transforms, *args, **kwargs)
        self.n_aug = n_aug

    def __call__(self, image, *args, **kwargs):
        image = torch.cat([
            super(ComposeMany, self).__call__(image=A.np.array(image), *args, **kwargs)['image']
            for _ in range(self.n_aug)
        ], dim=0)
        return image


class ComposeManyTorch(T.Compose):
    '''
    T.Compose derivative for sampling many augmentations of one image.
    '''

    def __init__(self, transforms, n_aug : int=1, *args, **kwargs):
        '''n_aug: number of augmentations to sample'''
        super().__init__(transforms, *args, **kwargs)
        self.n_aug = n_aug

    def __call__(self, image, *args, **kwargs):
        image = torch.cat([
            super(ComposeManyTorch, self).__call__(image, *args, **kwargs)
            for _ in range(self.n_aug)
        ], dim=0)
        return image


class MNISTDataModule2(MNISTDataModule):
    '''MNISTDataModule derivative to allow for a specified batch size.'''

    def __init__(self, batch_size: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def train_dataloader(self, batch_size=32, transforms=None):
        return super().train_dataloader(batch_size=self.batch_size, transforms=transforms)

    def val_dataloader(self, batch_size=32, transforms=None):
        return super().val_dataloader(batch_size=self.batch_size, transforms=transforms)

    def test_dataloader(self, batch_size=32, transforms=None):
        return super().test_dataloader(batch_size=self.batch_size, transforms=transforms)