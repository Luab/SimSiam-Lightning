import wandb


sweep_config = {
  "method": "bayes",  # bayes | random | grid
  "metric": {         # Maximize val_acc
      "name": "valid_acc",
      "goal": "maximize"
  },
  "parameters": {
        "num_layer_1": {  # Choose from pre-defined values
            "values": [32, 64, 128, 256, 512]
        },
        "num_layer_2": {
            "values": [32, 64, 128, 256, 512, 1024]
        },
        "lr": {  # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -9.21,   # exp(-9.21) = 1e-4
            "max": -4.61    # exp(-4.61) = 1e-2
        }
    }
}