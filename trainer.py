import os
import importlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from datetime import datetime
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional
from sklearn.preprocessing import OneHotEncoder

class Trainer():
    def __init__(self, data_dir: str, config_dir: str) -> None:
        self.data_dir = data_dir
        self.config_dir = config_dir
        
        self.config_modules = []
        self._load_module_from_path()

        self.configs = [module.config for module in self.config_modules]


    def _load_module_from_path(self):
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
            train_X_df = pd.read_csv(data['train_X']['path'], index_col=data['train_X']['index_col'])
            train_y_df = pd.read_csv(data['train_y']['path'], index_col=data['train_y']['index_col'])

            train_X = torch.tensor(train_X_df.to_numpy(dtype=np.float32))
            train_y = torch.tensor(train_y_df.to_numpy(dtype=np.float32))
            output_dir = data['output_dir']

            transform = data['transform'] if 'transform' in data else None
            if transform is not None:
                transform_steps = transform['steps']
                for step in transform_steps:
                    for transform_obj, params in step.items():
                        if transform_obj is pd.get_dummies:
                            train_X_df = pd.get_dummies(train_X_df, params["columns"])
                        else:
                            obj_instance = transform_obj(**params["params"])
                            targets = params["fit_transform_cols"]

                            if isinstance(obj_instance, OneHotEncoder):
                                ##TODO
                                train_X_df = getattr(obj_instance, 'fit_transform')(targets)
                            else:
                                train_X_df[targets] = getattr(obj_instance, 'fit_transform')(train_X_df[targets])

            hyperparameters = config['hyper_params']
            device = hyperparameters['device']
            print(f"running device: {device}")

            epochs = hyperparameters['epochs']
            optim = hyperparameters['optim']
            optim_params = hyperparameters['optim_params']
           
            data_loader_params = hyperparameters['data_loader_params']
            dataset = TensorDataset(train_X, train_y)
            dataloader = DataLoader(dataset, **data_loader_params)

            model = model_class(module_list).to(device)
            loss_func = hyperparameters['loss']
            optimizer = optim(model.parameters(), **optim_params)

            pbar = tqdm(range(epochs))
            for _ in pbar:
                trn_loss = self.train_one_epoch(model=model, dataloader=dataloader, loss_function=loss_func, optimizer=optimizer, device=device)
                pbar.set_postfix({"trn_loss": trn_loss})

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

            # TODO X, y 분포도 시각화.  TODO 스크린샷으로

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

            skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
            metrics: torchmetrics.MetricCollection = hyperparameters['metrics']
            metrics = metrics.to(device)
            history = defaultdict(list)

            for i, (trn_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
                X_trn, y_trn = train_X[trn_idx], train_y[trn_idx]
                X_val, y_val = train_X[val_idx], train_y[val_idx]

                ds_trn = TensorDataset(X_trn, y_trn)
                ds_val = TensorDataset(X_val, y_val)

                dl_trn = DataLoader(ds_trn, **data_loader_params)
                dl_val = DataLoader(ds_val, **data_loader_params)

                loss_func = hyperparameters['loss']
                optimizer = optim(models[i].parameters(), **optim_params)

                pbar = tqdm(range(epochs))
                for _ in pbar: 
                    trn_loss = self.train_one_epoch(model=models[i], dataloader=dl_trn ,loss_function=loss_func, optimizer=optimizer, device=device)
                    val_loss = self.validate_one_epoch(model=models[i], dataloader=dl_val, loss_function=loss_func, metrics=metrics, device=device)

                    history['trn_loss'].append(trn_loss)
                    history['val_loss'].append(val_loss)

                    result = metrics.compute()
                    for metric_name, metric_value in result.items():
                        history[metric_name].append(metric_value.item())
                    metrics.reset()
                    pbar.set_postfix({"acc": history['accuracy'][-1], "trn_loss": trn_loss, "val_loss": val_loss})
                    #TODO 시각화. loss 들 시각화. TODO 스크린샷으로
            
            #logging
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            output_dir += "/validation"
            output_dir = output_dir.replace("//", "/")

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(history['accuracy'], label='Accuracy')
            plt.title('Accuracy over Epochs')
            plt.ylabel('Accuracy', rotation=0)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(history['trn_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')

            plt.title('Training Progress')
            plt.ylabel('Value', rotation=0)
            plt.xlabel('Epoch')
            plt.legend()
            plt.tight_layout()

            file_name = f"{config_name}_history_{now}.png"
            plt.savefig(f"{output_dir}/{file_name}")
            # plt.show()

            df_metrics = pd.DataFrame(history)
            df_metrics = df_metrics.iloc[::epochs] #fold 단위로만 기록. 다남기고 싶으면 주석처리 or 파라미터화
            print(df_metrics)

            file_name = f"{config_name}_history_{now}.csv"
            df_metrics.to_csv(f"{output_dir}/{file_name}", index=False)

            file_name = output_dir + '/' + data['train_X']['path'].split('/')[-1]
            file_name = file_name.replace(".csv", "_summary.csv")
            if os.path.exists(file_name):
                df_summary = pd.read_csv(file_name)
            else:
                columns = [
                    'datetime',
                    'config_name', 
                    'model_name', 
                    'module_list', 
                    'loss', 
                    'optim', 
                    'lr', 
                    'epochs', 
                    'trn_loss_mean',
                    'trn_loss_std',
                    'val_loss_mean', 
                    'val_loss_std'
                ]
                for metric_name in history.keys():
                    if metric_name == 'trn_loss' or metric_name == 'val_loss':
                        continue
                    columns.append(metric_name + '_mean')
                    columns.append(metric_name + '_std')
                df_summary = pd.DataFrame(columns=columns)
            row = {
                'datetime': now,
                'config_name': config_name,
                'model_name': model_class.__name__,
                'module_list': self._convert_all_values_to_str(module_list),
                'loss': type(hyperparameters['loss']).__name__,
                'optim': optim.__name__,
                'lr': optim_params['lr'],
                'epochs': epochs,
                'trn_loss_mean': np.mean(history['trn_loss']),
                'trn_loss_std': np.std(history['trn_loss']),
                'val_loss_mean': np.mean(history['val_loss']),
                'val_loss_std': np.std(history['val_loss']),
            }
            for metric_name in history.keys():
                if metric_name == 'trn_loss' or metric_name == 'val_loss':
                    continue
                row[metric_name + '_mean'] = np.mean(history[metric_name])
                row[metric_name + '_std'] = np.std(history[metric_name])
            df_summary = pd.concat([df_summary, pd.DataFrame([row])])
            df_summary.to_csv(file_name, index=False)

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


    def train_one_epoch(self, model: nn.Module, dataloader: DataLoader, loss_function, optimizer: torch.optim.Optimizer, device) -> float:
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