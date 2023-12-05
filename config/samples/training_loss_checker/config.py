import torch.nn.functional as F
import torch
import torch.nn as nn

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from models.ann import ANN

config = {
    "name": "training_loss_checker",
    "model": {
        "class": ANN,
        "params": {},
    },
    "data": {
        "train_X": {
            "path": "data/sample_X.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "data/sample_y.csv",
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
        "loss": nn.BCELoss(),
        "optim": torch.optim.Adam,
        "optim_params": {
            "lr": 0.01,
        },
        "metrics": {
            'accuracy': BinaryAccuracy(),
            'precision' : BinaryPrecision(),
            'recall' : BinaryRecall(),
            'f1score' : BinaryF1Score(),
            'auroc' : BinaryAUROC(),
        },
        "device": "cuda"
        if torch.cuda.is_available()
        else "cpu",
        "epochs": 2000,
        'cv_params':{
            'n_split': 5,
        },
    },
}
