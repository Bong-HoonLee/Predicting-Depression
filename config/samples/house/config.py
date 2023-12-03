import torch
import torch.nn.functional as F
import torchmetrics
from models.ann import ANN

config = {
    "name": "MyConfig01",
    "model": {
        "class": ANN,
        "params": {
            "perceptron": (297, 128, 1),
            "dropout": 0.3,
            "activation": F.relu,
        },
    },
    "data": {
        "train_X": {
            "path": "data/trn_X.csv",
            "index_col": "Id"
        },
        "train_y": {
            "path": "data/trn_y.csv",
            "index_col": "Id"
        },
        "test_X": {
            "path": "data/tst_X.csv",
            "index_col": "Id"
        },
        "test_y": {
            "path": "data/tst_y.csv",
            "index_col": "Id"
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