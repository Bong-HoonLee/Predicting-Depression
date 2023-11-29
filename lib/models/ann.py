import pandas as pd
import numpy as np
import torch
import torchmetrics


import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from typing import Optional

class ANN(nn.Module):
    def __init__(
            self, 
            perceptron = (4, 128, 1),
            dropout=0.3,
            optim=torch.optim.Adam,
            loss_function=F.binary_cross_entropy,
            lr=1e-3,
            device=""
        ):
        super().__init__()

        input, hidden, output = perceptron
        self.lin1 = nn.Linear(input, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, output)
        self.dropout = nn.Dropout(dropout)
        
        self.optim = optim
        self.lr = lr
        self.loss_function: F = loss_function
        self.device = device
        if self.device == "":
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.optimizer = self.optim(self.parameters(), lr=self.lr)

        self.history = []


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = nn.functional.sigmoid(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = nn.functional.sigmoid(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = nn.functional.sigmoid(x)
        return x
    

    def training_step(self, data_loader: DataLoader) -> float:
        self.train()
        total_loss = 0.
        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.forward(X)
            loss = self.loss_function(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(y)
        return total_loss / len(data_loader.dataset)


    def validation_step(self, data_loader: DataLoader, metric:Optional[torchmetrics.metric.Metric]=None):
        self.eval()
        total_loss = 0.
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.forward(X)
                total_loss += self.loss_function(y_hat, y).item() * len(y)
                if metric is not None:
                    metric.update(y_hat, y)
        return total_loss / len(data_loader.dataset)
    
    def on_validation_epoch_end(self):
        pass


    def test_step(self, data_loader: DataLoader):
        pass