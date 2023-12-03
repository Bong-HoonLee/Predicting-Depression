import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_activation_stack = nn.Sequential(
            # 1
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            
            # 2
            nn.Linear(128, 1)
        )

    def forward(self, x):
        logits = self.linear_activation_stack(x)
        return logits