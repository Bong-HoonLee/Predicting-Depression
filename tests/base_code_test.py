# 빌트인, 써트파트 import
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAccuracy

src_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
sys.path.append(src_path)

dir_list = os.listdir(src_path)
for item in dir_list:
    path = src_path + "/" + item
    if not os.path.isdir(path):
        continue
    if item.startswith("."):
        continue
    if path in sys.path:
        continue
    sys.path.append(path)

from lib.models.ann import ANN

# 하이퍼 파라미터
N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 42
DROP_OUT = 0.3
KFOLD_SHUFFLE = False
EHPOCHS = 100

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# 데이터 준비 & 전처리
root_dir = os.getcwd()
train_df = pd.read_csv(f"{root_dir}/tests/training_data_202311291425.csv")

train_df_without_y = train_df.drop(columns=["DF2_pr"])

X_train = pd.get_dummies(train_df_without_y).to_numpy(dtype=np.float32)
y_train = train_df[["DF2_pr"]].to_numpy(dtype=np.float32)


# 모델, 손실함수, 옵티마이저 준비
nets = [
        ANN(
            perceptron=(284, 512, 1), 
            optim=torch.optim.Adam, 
            loss_function=F.binary_cross_entropy,
            dropout=DROP_OUT,
            lr=LEARNING_RATE,
            device=device
        ).to(device) for _ in range(N_SPLITS)
    ]
history = []

# 학습
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=KFOLD_SHUFFLE, random_state=None)

for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X, y = torch.tensor(X_train[trn_idx]), torch.tensor(y_train[trn_idx])
    X_val, y_val = torch.tensor(X_train[val_idx]), torch.tensor(y_train[val_idx])

    ds = TensorDataset(X, y) 
    ds_val = TensorDataset(X_val, y_val)

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

    net = nets[i]
    optimizer = nets[i].optimizer

    # 교차평가
    pbar = tqdm(range(EPOCHS))
    for j in pbar:
        accuracy = BinaryAccuracy().to(device)
        loss = net.training_step(data_loader=dl)
        loss_val = net.validation_step(data_loader=dl_val, metric=accuracy)
        acc_val = accuracy.compute().item()
        pbar.set_postfix(trn_loss=loss, val_loss=loss_val, val_acc=acc_val)

    bcm = BinaryConfusionMatrix().to(device)
    net.validation_step(data_loader=dl_val, metric=bcm)
    history.append(bcm)

# 교차평가 계속(시각화) 
cm = sum([bcm.compute().cpu().numpy() for bcm in history])
ConfusionMatrixDisplay(cm).plot()
plt.show()

(tn, fp), (fn, tp) = cm
accuracy = (tp+tn)/(tp+tn+fn+fp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*(precision*recall)/(precision+recall)

pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1]})