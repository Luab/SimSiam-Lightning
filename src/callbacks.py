# callbacks.py

import torch
from torch.nn import functional as F

from pytorch_lightning.callbacks import Callback

import wandb


class ImagePredictionLogger(Callback):
    '''Modified from
    https://colab.research.google.com/drive/12oNQ8XGeJFMiSBGsQ8uth8TaghVBC9H3'''
    def __init__(self, val_samples, n_samples=32):
        super().__init__()
        self.n_samples = n_samples
        self.val_x, self.val_y = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_x = self.val_x.to(device=pl_module.device)
        val_y = self.val_y.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_x)
        probs = torch.max(F.softmax(logits, dim=1), -1).values
        #probs = torch.max(probs, -1).values
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"P({pred})={prob:.2f}, Label:{y}") 
                        for x, pred, prob, y in zip(val_x[:self.n_samples],
                                                    preds[:self.n_samples],
                                                    probs[:self.n_samples],
                                                    val_y[:self.n_samples])]
        })