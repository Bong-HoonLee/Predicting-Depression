import torch.nn as nn
import torchmetrics
import torch

from torch.utils.data import DataLoader

class Modelable():
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def training_step(self, dataloader: DataLoader, loss_function, optimizer: torch.optim.Optimizer, device, metric: torchmetrics.metric.Metric) -> float:
        pass

    def validation_step(self, dataloader: DataLoader, device, metric: torchmetrics.metric.Metric):
        pass
    
    def test_step(self, dataloader: DataLoader):
        pass