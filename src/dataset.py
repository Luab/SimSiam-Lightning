import albumentations as A
import torch


class ComposeMany(A.Compose):
    '''
    Class derived from A.Compose for sampling many augmentations of one image.
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