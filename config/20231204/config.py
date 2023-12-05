import torch
import torch.nn as nn
import torchmetrics

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from models.ann import ANN

config = {
    "name": "HN_X_231205",
    "model": {
        "class": ANN,
        "module_list": nn.ModuleList([
            nn.Linear(224, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ]),
    },
    "data": {
        "train_X": {
            "path": "data/HN_X_231205.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "data/HN_y_231205.csv",
            "index_col": None,
        },
        "test_X": {
            "path": "data/HN_test_X.csv",
            "index_col": None,
        },
        "test_y": {
            "path": "data/HN_test_y.csv",
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
        "metrics": torchmetrics.MetricCollection({
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'f1score': BinaryF1Score(),
            'auroc': BinaryAUROC(),
            'mse': torchmetrics.MeanSquaredError(),
        }),
        "device": "cuda"
        if torch.cuda.is_available()
        else "cpu",
        "epochs": 100,
        'cv_params':{
            'n_split': 5,
        },
    },
}
