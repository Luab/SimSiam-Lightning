# utils.py

def sweep_iteration(proj):
    # set up W&B logger
    wandb.init()    # required to have access to `wandb.config`
    wandb_logger = WandbLogger()

    # setup data
    mnist = MNISTDataModule('../data/')

    # setup model - note how we refer to sweep parameters with wandb.config
    mlp = MLP(
        num_classes=mnist.num_classes, C=C, W=W, H=H,
        num_layer_1=wandb.config.num_layer_1,
        num_layer_2=wandb.config.num_layer_2,
    )
    model = LitModel(datamodule=mnist, arch=mlp, lr=wandb.config.lr)

    # setup Trainer
    trainer = Trainer(
        logger=wandb_logger,    # W&B integration
        #gpus=-1,                # use all GPU's
        max_epochs=3            # number of epochs
    )

    # train
    trainer.fit(model, mnist)