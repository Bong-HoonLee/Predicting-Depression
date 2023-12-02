import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  Accuracy:torchmetrics.metric.Metric,
  Precision:torchmetrics.metric.Metric,
  Recall:torchmetrics.metric.Metric,
  F1Score:torchmetrics.metric.Metric,
  AUROC:torchmetrics.metric.Metric,
  device:str='cpu',
) -> None:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
      metrcis: metrics
  '''
  model.eval()
  total_loss = 0.
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)
      Accuracy.update(output, y)
      Precision.update(output, y)
      Recall.update(output, y)
      F1Score.update(output, y)
      AUROC.update(output, y)

      return total_loss/len(data_loader.dataset)

