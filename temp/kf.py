import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Type
import pandas as pd

@dataclass
class KFoldCV:
  X: torch.Tensor
  y: torch.Tensor
  Model: Type[nn.Module]
  input_dim: int
  epochs: int = 500
  criterion: callable = nn.BCEWithLogitsLoss
  Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
  optim_kwargs: dict = field(default_factory=lambda : {})
  trn_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 36})
  val_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 36})
  n_splits: int = 5
  metric: dict = field(default_factory=lambda : {})
  device: str = 'cpu'

  def run(self):
    from torch.utils.data import TensorDataset
    from sklearn.model_selection import KFold
    from tqdm.auto import trange
    from train import train_one_epoch
    from eval import evaluate

    model = self.Model(self.input_dim).to(self.device)
    models = [self.Model(self.input_dim).to(self.device) for _ in range(self.n_splits)]
    for m in models:
      m.load_state_dict(model.state_dict())
    kfold = KFold(n_splits=self.n_splits, shuffle=False)

    # metric setting
    accuracy_met = self.metric.get('accuracy')
    precision_met = self.metric.get('precision')
    recall_met = self.metric.get('recall')
    f1score_met = self.metric.get('f1score')
    auroc_met = self.metric.get('auroc')

    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1Score': [], 'AUROC': []}
    for i, (trn_idx, val_idx) in enumerate(kfold.split(self.X)):
      X_trn, y_trn = self.X[trn_idx], self.y[trn_idx]
      X_val, y_val = self.X[val_idx], self.y[val_idx]

      ds_trn = TensorDataset(X_trn, y_trn)
      ds_val = TensorDataset(X_val, y_val)

      dl_trn = DataLoader(ds_trn, **self.trn_dl_kwargs)
      dl_val = DataLoader(ds_val, **self.val_dl_kwargs)

      m = models[i]
      optim = self.Optimizer(m.parameters(), **self.optim_kwargs)

      pbar = trange(self.epochs)
      for _ in pbar:
        # train
        trn_loss = train_one_epoch(m, self.criterion, optim, dl_trn, self.device)

        # metric reset
        accuracy_met.reset()
        precision_met.reset()
        recall_met.reset()
        f1score_met.reset()
        auroc_met.reset()

        # eval
        val_loss = evaluate(m, self.criterion, dl_val, accuracy_met, precision_met, recall_met, f1score_met, auroc_met, self.device)

        # metric
        Accuracy = accuracy_met.compute().item()
        Precision = precision_met.compute().item()
        Recall = recall_met.compute().item()
        F1Score = f1score_met.compute().item()
        AUROC = auroc_met.compute().item()

        pbar.set_postfix(trn_loss=trn_loss, val_loss=val_loss)
      metrics['Accuracy'].append(Accuracy)
      metrics['Precision'].append(Precision)
      metrics['Recall'].append(Recall)
      metrics['F1Score'].append(F1Score)
      metrics['AUROC'].append(AUROC)
    return pd.DataFrame(metrics)

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  import numpy as np
  from nn import ANN

  args = get_args_parser().parse_args()
  
  exec(open(args.config).read())
  cfg = config

  train_params = cfg.get('train_params')
  device = train_params.get('device')

  files = cfg.get('files')
  X_df = pd.read_csv(files.get('X_csv'))
  y_df = pd.read_csv(files.get('y_csv'))

  X, y = torch.tensor(X_df.to_numpy(dtype=np.float32)), torch.tensor(y_df.to_numpy(dtype=np.float32)).reshape(-1, 1)
  
  # model setting
  Model = cfg.get('model')
  input_dim = X.shape[-1]
  
  dl_params = train_params.get('data_loader_params')

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')

  cv_params = cfg.get('cv_params')
  n_splits = cv_params.get('n_split')

  # metrics setting
  metric = train_params.get('metric')

  cv = KFoldCV(X, y, Model, input_dim=input_dim,
               epochs=train_params.get('epochs'),
               criterion=train_params.get('loss_fn'),
               Optimizer=Optim,
               optim_kwargs=optim_params,
               trn_dl_kwargs=dl_params, val_dl_kwargs=dl_params,
               n_splits=n_splits,
               metric=metric,
               device=device)
  res = cv.run()

  res = pd.concat([res, res.apply(['mean', 'std'])])
  print(res)
  res.to_csv(files.get('output_csv'))