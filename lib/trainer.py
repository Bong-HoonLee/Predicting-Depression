import pandas as pd
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from datetime import datetime

from lib.models.factory import ModelBuilder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAccuracy

class Trainer():
    def __init__(self, params: tuple) -> None:
        self.user = params.user
        self.mode = params.mode
        self.device = params.device
        self.model = params.model
        self.model_config_file = params.model_config_file
        self.model_output_dir = params.model_output_dir
        self.train_data_file = params.train_data_file

        self._train_df = pd.read_csv(self.train_data_file)

        self._model = ModelBuilder.make(
            model=params.model, 
            params=params
        )

        self._history = []

    #TODO 하드코딩 남아 있는 부분 -> 하이퍼파라미터로 변경
    def fit(self, kfold_n_splits: int=0):
        # 학습
        if kfold_n_splits == 0:
            pass
        else: #교차검증
            nets = [
                copy.deepcopy(self._model) for _ in range(kfold_n_splits)
            ]

            self._train_df['Sex'] = self._train_df['Sex'].map({'female': 0, 'male': 1})

            X_train = pd.get_dummies(self._train_df[["Pclass", "Sex", "SibSp", "Parch"]]).to_numpy(dtype=np.float32)
            y_train = self._train_df[["Survived"]].to_numpy(dtype=np.float32)

            skf = StratifiedKFold(n_splits=kfold_n_splits, shuffle=False, random_state=None)

            for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X, y = torch.tensor(X_train[trn_idx]), torch.tensor(y_train[trn_idx])
                X_val, y_val = torch.tensor(X_train[val_idx]), torch.tensor(y_train[val_idx])

                ds = TensorDataset(X, y) 
                ds_val = TensorDataset(X_val, y_val)

                dl = DataLoader(ds, batch_size=32, shuffle=True)
                dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

                net = nets[i]
                optimizer = nets[i].optimizer

                # 교차평가
                pbar = tqdm(range(100))
                for j in pbar:
                    accuracy = BinaryAccuracy().to(self.device)
                    loss = net.training_step(data_loader=dl)
                    loss_val = net.validation_step(data_loader=dl_val, metric=accuracy)
                    acc_val = accuracy.compute().item()
                    pbar.set_postfix(trn_loss=loss, val_loss=loss_val, val_acc=acc_val)

                bcm = BinaryConfusionMatrix().to(self.device)
                net.validation_step(data_loader=dl_val, metric=bcm)
                self._history.append(bcm)
                

            cm = sum([bcm.compute().cpu().numpy() for bcm in self._history])
            ConfusionMatrixDisplay(cm).plot()
            plt.show()

            (tn, fp), (fn, tp) = cm
            accuracy = (tp+tn)/(tp+tn+fn+fp)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2*(precision*recall)/(precision+recall)

            metrics = pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1]})
            
            metrics_file = "metrics_" + datetime.now().strftime("%Y%m%d%H%M%S")
            metrics.to_csv(f"{self.model_output_dir}/{metrics_file}.csv", index=False)

            #TODO step 별 logging?

    def validate(self):
        # self.model.validate()
        pass

    def test(self):
        # self.model.test()
        pass