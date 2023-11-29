import torch
import torch.nn.functional as F

class HyperParams:
    BATCH_SIZE = 32
    EPOCHS = 1000
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    RANDOM_STATE = 42
    DROP_OUT = 0.3
    KFOLD_N_SPLITS = 5
    KFOLD_SHUFFLE = False
    EHPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class ModelParams:
    perceptron = (4, 128, 1)
    optim = torch.optim.Adam
    loss_function = F.binary_cross_entropy
    dropout = 0.3
    lr = 1e-3