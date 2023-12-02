import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from nn import ANN

config = {

  'files': {
    'X_csv': './data/trn_X.csv',
    'y_csv': './data/trn_y.csv',
    'output': './model.pth',
    'output_csv': './results/five_fold.csv',
  },

  'model': ANN,
  'model_params': {
    'input_dim': 'auto' # Always will be determined by the data shape
  },

  'train_params': {
    'data_loader_params': {
      'batch_size': 32,
      'shuffle': True
    },
    'loss_fn': nn.BCEWithLogitsLoss(),
    'optim': torch.optim.Adam,
    'optim_params': {
      'lr': 0.01,
    },
    'metric': {'accuracy': BinaryAccuracy(),
               'precision' : BinaryPrecision(),
               'recall' : BinaryRecall(),
               'f1score' : BinaryF1Score(),
               'auroc' : BinaryAUROC()
               },
    'device': 'cpu',
    'epochs': 300,
  },

  'cv_params':{
    'n_split': 5,
  },

}