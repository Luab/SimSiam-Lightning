## losses.py
# TODO: tests

import torch
from torch.nn import functional as F


def flood(x, flood_height : float = 0.05):
    '''Flood the loss value (see arXiv paper).'''
    return torch.abs(x - flood_height) + flood_height


def log_softmax(batch, forward_callable):
    '''Cross-entropy loss.'''
    x, y = batch
    logits = forward_callable(x)
    log_y_hat = F.log_softmax(logits, dim=1)  # log probability
    return F.nll_loss(log_y_hat, y)


def stopgrad(x):
    return x.detach()


# BUG? dim=2
def negcosim(p, z):
    '''Negative cosine similarity.'''
    p = F.normalize(p, dim=1)  # l2-normalize
    z = F.normalize(z, dim=1)
    return -(p*z).sum(dim=1).mean(dim=0)


def simsiam_loss(batch, forward_callable):
    x, y = batch
    z1, z2, p1, p2 = forward_callable(x)
    return (negcosim(p1, z2) + negcosim(p2, z1)) / 2
    #return (negcosim(p1, stopgrad(z2)) + negcosim(p2, stopgrad(z1))) / 2