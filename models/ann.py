import torch
import torchmetrics

import torch.nn as nn

from torch.utils.data import DataLoader
from models.modelable import Modelable

class ANN(Modelable, nn.Module):
    def __init__(
            self, 
            perceptron = (4, 128, 1),
            dropout=0.3,
            activation=nn.functional.relu,
        ):
        super().__init__()

        input, hidden, output = perceptron
        self.lin1 = nn.Linear(input, hidden)
        self.lin2 = nn.Linear(hidden, output)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        self.history = []


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x
    

    def training_step(self, dataloader: DataLoader, loss_function, optimizer: torch.optim.Optimizer, device, metric: torchmetrics.metric.Metric) -> float:
        self.train()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = self.forward(X)
            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metric.update(y_hat, y)

    def validation_step(self, dataloader: DataLoader, device, metric: torchmetrics.metric.Metric):
        self.eval()
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_hat = self.forward(X)
                metric.update(y_hat, y)

    def test_step(self, dataloader: DataLoader):
        pass