import torch.nn.functional as F
import torch
import torchmetrics
from models.ann import ANN

config = {
    "name": "MyConfig01",
    "model": {
        "class": ANN,
        "params": {
            "perceptron": (4, 128, 1),
            "dropout": 0.1,
            "activation": F.relu,
        },
    },
    "data": {
        "train_X": {
            "path": "data/train_X.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "data/train_y.csv",
            "index_col": None,
        },
        "test_X": {
            "path": "data/test_X.csv",
            "index_col": None,
        },
        "test_y": {
            "path": "data/test_y.csv",
            "index_col": None,
        },
        "output_dir": "output/",
    },
    "hyper_params": {
        "data_loader_params": {
            "batch_size": 32,
            "shuffle": True,
        },
        "loss": F.mse_loss,
        "optim": torch.optim.Adam,
        "optim_params": {
            "lr": 0.01,
        },
        "metric": torchmetrics.MeanSquaredError(squared=False),
        "device": "cuda"
        if torch.cuda.is_available()
        else "cpu",
        "epochs": 300,
        'cv_params':{
            'n_split': 5,
        },
    },
}