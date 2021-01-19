# utils.py

import torch
from torch.nn import functional as F

from tqdm import tqdm

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

from src.models import CNN, BaseLitModel


def sweep_iteration(proj):
    wandb.init()    # required to have access to `wandb.config`
    wandb_logger = WandbLogger()
    mnist = MNISTDataModule('../data/')
    # setup model - note how we refer to sweep parameters with wandb.config
    mlp = MLP(
        num_classes=mnist.num_classes, C=C, W=W, H=H,
        num_layer_1=wandb.config.num_layer_1,
        num_layer_2=wandb.config.num_layer_2,
    )
    model = LitModel(datamodule=mnist, arch=mlp, lr=wandb.config.lr)
    trainer = Trainer(
        logger=wandb_logger,    # W&B integration
        #gpus=-1,                # use all GPU's
        max_epochs=3            # number of epochs
    )
    trainer.fit(model, mnist)


def mnist_data():
    mnist = MNISTDataModule('../data/', batch_size=512)
    mnist.prepare_data()
    mnist.setup()
    return mnist


def lit_model():
    mnist = mnist_data()
    cnn = CNN(num_channels=mnist.dims[0], num_classes=mnist.num_classes)  # Architecture
    return BaseLitModel(datamodule=mnist, backbone=cnn, lr=1e-3, flood_height=0)  # Lightning model


def knn_monitor(model, memory_dataloader, test_dataloader,
                device, epoch, knn_k, knn_t=1, epochs=''):
    '''Test using a k-nn monitor.'''

    model.eval()
    classes = len(memory_dataloader.dataset.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []

    with torch.no_grad():
        ## Generate feature bank.
        for data, _ in tqdm(memory_dataloader, desc='Feature extracting'):
            feature = model(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        ## [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        ## [N]
        feature_labels = memory_dataloader.dataset.dataset.targets.to(feature_bank.device)
        ## Loop test data to predict the label by weighted knn search.
        test_bar = tqdm(test_dataloader)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels,
                                      classes, knn_k, knn_t)
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(
                    epoch, epochs, total_top1 / total_num * 100
                )
            )
    return total_top1 / total_num * 100


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t=1):
    '''
    k-NN monitor as in InstDisc https://arxiv.org/abs/1805.01978
    Implementation follows http://github.com/zhirongw/lemniscate.pytorch
    and https://github.com/leftthomas/SimCLR
    '''
    ## Cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    ## [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    ## [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    ## Counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    ## [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    ## Weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels