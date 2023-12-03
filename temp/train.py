import torch
from torch import nn
from torch.utils.data import DataLoader

def train_one_epoch(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(y)
  return total_loss/len(data_loader.dataset)


def main(cfg):
  import numpy as np
  import pandas as pd
  from torch.utils.data.dataset import TensorDataset
  from nn import ANN
  from tqdm.auto import trange

  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  
  files = cfg.get('files')
  X_trn = torch.tensor(pd.read_csv(files.get('X_csv')).to_numpy(dtype=np.float32))
  y_trn = torch.tensor(pd.read_csv(files.get('y_csv')).to_numpy(dtype=np.float32)).reshape(-1, 1)

  dl_params = train_params.get('data_loader_params')
  ds = TensorDataset(X_trn, y_trn)
  dl = DataLoader(ds, **dl_params)

  Model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = X_trn.shape[-1]
  model = Model(**model_params).to(device)

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')
  optimizer = Optim(model.parameters(), **optim_params)

  loss_fn = train_params.get('loss_fn')
  pbar = trange(train_params.get('epochs'))
  for _ in pbar:
    loss = train_one_epoch(model, loss_fn, optimizer, dl, device)
    pbar.set_postfix(loss=loss)
  torch.save(model.state_dict(), files.get('output'))

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config)