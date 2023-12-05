import os
import importlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchmetrics

from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional

class Trainer():
    def __init__(self, data_dir: str, config_dir: str) -> None:
        self.data_dir = data_dir
        self.config_dir = config_dir
        
        self.config_modules = []
        self.load_module_from_path()

        self.configs = [module.config for module in self.config_modules]


    def load_module_from_path(self):
        config_files = os.listdir(self.config_dir)
        for config_file in config_files:
            config_file = os.path.join(self.config_dir, config_file)
            module_name = os.path.splitext(os.path.basename(config_file))[0]
            spec = importlib.util.spec_from_file_location(module_name, config_file)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.config_modules.append(module)
        

    def fit(self):
        for config in self.configs:
            model = config['model']
            model_class = model['class']
            module_list = model['module_list']

            data = config['data']
            train_X = torch.tensor(pd.read_csv(data['train_X']['path'], index_col=data['train_X']['index_col']).to_numpy(dtype=np.float32))
            train_y = torch.tensor(pd.read_csv(data['train_y']['path'], index_col=data['train_y']['index_col']).to_numpy(dtype=np.float32))
            output_dir = data['output_dir']

            hyperparameters = config['hyper_params']
            device = hyperparameters['device']
            print(f"running device: {device}")

            epochs = hyperparameters['epochs']
            optim = hyperparameters['optim']
            optim_params = hyperparameters['optim_params']
            metrics: torchmetrics.MetricCollection = hyperparameters['metrics']
           
            data_loader_params = hyperparameters['data_loader_params']
            dataset = TensorDataset(train_X, train_y)
            dataloader = DataLoader(dataset, **data_loader_params)

            model = model_class(module_list).to(device)
            loss_func = hyperparameters['loss']
            optimizer = optim(model.parameters(), **optim_params)

            pbar = tqdm(range(epochs))
            for _ in pbar:
                _ = self.train_one_epoch(model=model, dataloader=dataloader, loss_function=loss_func, optimizer=optimizer, device=device, metrics=metrics)
                mse = metrics['mse'].compute().item()
                metrics.reset()
                pbar.set_postfix({"mse": mse, "rmse": np.sqrt(mse)})

            file_name = config["name"] + "_" + datetime.now().strftime("%Y%m%d%H%M") + ".pth"
            torch.save(model.state_dict(), f"{output_dir}/{file_name}")


    def validate(self):
        for config in self.configs:
            config_name = config['name']
            model = config['model']
            model_class = model['class']
            module_list = model['module_list']

            data = config['data']
            train_X = torch.tensor(pd.read_csv(data['train_X']['path'], index_col=data['train_X']['index_col']).to_numpy(dtype=np.float32))
            train_y = torch.tensor(pd.read_csv(data['train_y']['path'], index_col=data['train_y']['index_col']).to_numpy(dtype=np.float32))
            output_dir = data['output_dir']

            hyperparameters = config['hyper_params']
            device = hyperparameters['device']
            print(f"running device: {device}")

            epochs = hyperparameters['epochs']
            optim = hyperparameters['optim']
            optim_params = hyperparameters['optim_params']

            n_split = hyperparameters['cv_params']['n_split']
            data_loader_params = hyperparameters['data_loader_params']

            model = model_class(module_list).to(device)
            models = [model_class(module_list).to(device) for _ in range(n_split)]
            for i, _ in enumerate(models):
                models[i].load_state_dict(model.state_dict())

            kfold = KFold(n_splits=n_split, shuffle=False)
            metrics: torchmetrics.MetricCollection = hyperparameters['metrics']
            history = []

            for i, (trn_idx, val_idx) in enumerate(kfold.split(train_X)):
                X_trn, y_trn = train_X[trn_idx], train_y[trn_idx]
                X_val, y_val = train_X[val_idx], train_y[val_idx]

                ds_trn = TensorDataset(X_trn, y_trn)
                ds_val = TensorDataset(X_val, y_val)

                dl_trn = DataLoader(ds_trn, **data_loader_params)
                dl_val = DataLoader(ds_val, **data_loader_params)

                loss_func = hyperparameters['loss']
                optimizer = optim(models[i].parameters(), **optim_params)

                history = defaultdict(list)

                pbar = tqdm(range(epochs))
                for _ in pbar:
                    _ = self.train_one_epoch(model=models[i], dataloader=dl_trn ,loss_function=loss_func, optimizer=optimizer, device=device)
                    _ = self.validate_one_epoch(model=models[i], dataloader=dl_val, loss_function=loss_func, metrics=metrics, device=device)
                    
                    result = metrics.compute()
                    for metric_name, metric_value in result.items():
                        history[metric_name].append(metric_value.item())
                    metrics.reset()
                    pbar.set_postfix({"mse": history['mse'][-1], "rmse": np.sqrt(history['mse'][-1])})

            #logging
            df_metrics = pd.DataFrame(history)
            print(df_metrics)

            output_dir += "/validation"
            output_dir = output_dir.replace("//", "/")
            datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"{config_name}_history_{datetime_str}.csv"
            df_metrics.to_csv(f"{output_dir}/{file_name}", index=False)

            file_name = output_dir + '/' + data['train_X']['path'].split('/')[-1]
            file_name = file_name.replace(".csv", "_summary.csv")
            if os.path.exists(file_name):
                df_meta = pd.read_csv(file_name)
            else:
                df_meta = pd.DataFrame(columns=[
                        'datetime',
                        'config_name', 
                        'model_name', 
                        'module_list', 
                        'loss', 
                        'optim', 
                        'lr', 
                        'metrics', 
                        'cv_n_split', 
                        'epochs', 
                        'val_loss_mean', 
                        'val_loss_std'
                    ])
            df_meta = pd.concat([df_meta, pd.DataFrame([{
                'datetime': datetime_str,
                'config_name': config_name,
                'model_name': model_class.__name__,
                'module_list': self._convert_all_values_to_str(module_list),
                'loss': type(hyperparameters['loss']).__name__,
                'optim': optim.__name__,
                'lr': optim_params['lr'],
                'metrics': ",".join([key for key in metrics.keys()]),
                'cv_n_split': n_split,
                'epochs': epochs,
                'val_loss_mean': np.mean(history['mse']),
                'val_loss_std': np.std(history['mse']),
            }])])
            df_meta.to_csv(file_name, index=False)


    def _convert_all_values_to_str(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_all_values_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [self._convert_all_values_to_str(elem) for elem in obj]
        else:
            return getattr(obj, '__name__', str(obj))


    def generate_sample(self):
        from sklearn.datasets import make_classification

        # 합성 데이터셋 생성
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, n_clusters_per_class=1, random_state=42)

        # NumPy 배열을 PyTorch 텐서로 변환
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        X_np = X.numpy()
        y_np = y.numpy()

        df = pd.DataFrame(X_np)
        df.to_csv("data/sample_X.csv", index=False)
        df = pd.DataFrame(y_np)
        df.to_csv("data/sample_y.csv", index=False)

    def predict(self):
        #TODO testset
        #TODO testset 고르게 추출. 나이 분포, 성별 분포 고려해서 분리. PHQ 점수 분포도 잘 맞추는지 테스트 필요(고르게 추출).
        pass


    def train_one_epoch(self, model: nn.Module, dataloader: DataLoader, loss_function, optimizer: torch.optim.Optimizer, device, metrics: Optional[torchmetrics.MetricCollection]=None) -> float:
        model.train()
        total_loss = 0.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model.forward(X)
            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            if metrics is not None:
                metrics.update(y_hat, y)
        return total_loss/len(dataloader.dataset)

    def validate_one_epoch(self, model: nn.Module, dataloader: DataLoader, loss_function, device, metrics: Optional[torchmetrics.MetricCollection]=None):
        model.eval()
        total_loss = 0.
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_hat = model.forward(X)
                total_loss += loss_function(y_hat, y).item() * len(y)
                if metrics is not None:
                    metrics.update(y_hat, y)
        return total_loss/len(dataloader.dataset)

    def test_step(self, model: nn.Module, dataloader: DataLoader):
        pass