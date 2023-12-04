import torch


import torch.nn as nn

#TODO multitask neural network
#TODO y 라벨 있는거에 대해서 학습
class ANN222(nn.Module):
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