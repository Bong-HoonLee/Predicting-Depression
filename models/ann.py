import torch.nn as nn
from copy import deepcopy


class ANN(nn.Module):
    def __init__(self, module_list: nn.ModuleList = []):
        super(ANN, self).__init__()
        self.module_list = deepcopy(module_list)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x