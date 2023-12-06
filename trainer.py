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
    def __init__(self, data_dir: str, config_dir: str, config_name: str = None) -> None:
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.config_name = config_name
        
        self.config_modules = []
        self._load_module_from_path()

        self.configs = [module.config for module in self.config_modules]
        if config_name is not None:
            self.configs = [config for config in self.configs if config['name'] == config_name]


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
        

    def train(self):
        for config in self.configs:
            model = config['model']
            model_class = model['class']
            module_list = model['module_list']

            data = config['data']
            cached_X_file = f"{self.data_dir}/transformed/{data['train_X']['path'].split('/')[-1]}"
            cached_y_file = f"{self.data_dir}/transformed/{data['train_y']['path'].split('/')[-1]}"
            if os.path.exists(cached_X_file) and os.path.exists(cached_y_file):
                train_X_df = pd.read_csv(cached_X_file)
                train_y_df = pd.read_csv(cached_y_file)
            else:
                train_X_df = pd.read_csv(data['train_X']['path'], index_col=data['train_X']['index_col'])
                train_y_df = pd.read_csv(data['train_y']['path'], index_col=data['train_y']['index_col'])

                transform = data['transform'] if 'transform' in data else None
                if transform is not None:
                    transform_steps = transform['steps']
                    for step in transform_steps:
                        for transform_obj, params in step.items():
                            obj_instance = transform_obj(**params["params"])
                            if obj_instance.__class__ == OneHotEncoder:
                                targets = params["fit_transform_cols"]
                                # 원-핫 인코딩 적용할 컬럼 선택
                                onehot_encoded = obj_instance.fit_transform(train_X_df[targets])
                                # 원-핫 인코딩된 데이터프레임 생성
                                onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=obj_instance.get_feature_names_out(targets))
                                # 원래 데이터프레임에서 인코딩 대상 컬럼 제거
                                train_X_df = train_X_df.drop(targets, axis=1)
                                # 원-핫 인코딩된 데이터프레임과 원래 데이터프레임 결합
                                train_X_df = pd.concat([train_X_df, onehot_encoded_df], axis=1)
                            elif hasattr(obj_instance, 'fit_resample'):
                                targets = params["fit_resample_cols"]
                                train_X_df, train_y_df = getattr(obj_instance, 'fit_resample')(train_X_df, train_y_df)
                            elif hasattr(obj_instance, 'fit_transform'):
                                targets = params["fit_transform_cols"]
                                train_X_df[targets] = getattr(obj_instance, 'fit_transform')(train_X_df[targets])
                            else:
                                raise ValueError("transform object must have fit_transform or fit_resample method")
                print(f"transformed train_X_df.shape: {train_X_df.shape}")
                print(f"transformed train_y_df.shape: {train_y_df.shape}")
                # train_X_df.to_csv(cached_X_file, index=False)
                # train_y_df.to_csv(cached_y_file, index=False)

            train_X = torch.tensor(train_X_df.to_numpy(dtype=np.float32))
            train_y = torch.tensor(train_y_df.to_numpy(dtype=np.float32))
            output_dir = data['output_dir']      
            
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
            cached_X_file = f"{self.data_dir}/transformed/{data['train_X']['path'].split('/')[-1]}"
            cached_y_file = f"{self.data_dir}/transformed/{data['train_y']['path'].split('/')[-1]}"
            if os.path.exists(cached_X_file) and os.path.exists(cached_y_file):
                train_X_df = pd.read_csv(cached_X_file)
                train_y_df = pd.read_csv(cached_y_file)
            else:
                train_X_df = pd.read_csv(data['train_X']['path'], index_col=data['train_X']['index_col'])
                train_y_df = pd.read_csv(data['train_y']['path'], index_col=data['train_y']['index_col'])

                transform = data['transform'] if 'transform' in data else None
                if transform is not None:
                    transform_steps = transform['steps']
                    for step in transform_steps:
                        for transform_obj, params in step.items():
                            obj_instance = transform_obj(**params["params"])
                            if obj_instance.__class__ == OneHotEncoder:
                                targets = params["fit_transform_cols"]
                                # 원-핫 인코딩 적용할 컬럼 선택
                                onehot_encoded = obj_instance.fit_transform(train_X_df[targets])
                                # 원-핫 인코딩된 데이터프레임 생성
                                onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=obj_instance.get_feature_names_out(targets))
                                # 원래 데이터프레임에서 인코딩 대상 컬럼 제거
                                train_X_df = train_X_df.drop(targets, axis=1)
                                # 원-핫 인코딩된 데이터프레임과 원래 데이터프레임 결합
                                train_X_df = pd.concat([train_X_df, onehot_encoded_df], axis=1)
                            elif hasattr(obj_instance, 'fit_resample'):
                                targets = params["fit_resample_cols"]
                                train_X_df, train_y_df = getattr(obj_instance, 'fit_resample')(train_X_df, train_y_df)
                            elif hasattr(obj_instance, 'fit_transform'):
                                targets = params["fit_transform_cols"]
                                train_X_df[targets] = getattr(obj_instance, 'fit_transform')(train_X_df[targets])
                            else:
                                raise ValueError("transform object must have fit_transform or fit_resample method")
                print(f"transformed train_X_df.shape: {train_X_df.shape}")
                print(f"transformed train_y_df.shape: {train_y_df.shape}")
                train_X_df.to_csv(cached_X_file, index=False)
                train_y_df.to_csv(cached_y_file, index=False)

            train_X = torch.tensor(train_X_df.to_numpy(dtype=np.float32))
            train_y = torch.tensor(train_y_df.to_numpy(dtype=np.float32))
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
            
            #logging
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            output_dir += "/validation"
            output_dir = output_dir.replace("//", "/")

            def divide_into_segments(data, segments):
                n = len(data)
                return [data[i * n // segments: (i + 1) * n // segments] for i in range(segments)]

            accuracy_segments = divide_into_segments(history['accuracy'], n_split)
            trn_loss_segments = divide_into_segments(history['trn_loss'], n_split)
            val_loss_segments = divide_into_segments(history['val_loss'], n_split)

            plt.figure(figsize=(10, 6))

            plt.subplot(2, 1, 1)
            for i, segment in enumerate(accuracy_segments):
                plt.plot(segment, label=f'Accuracy Fold {i + 1}')
            plt.title('Accuracy over Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(2, 1, 2)
            for i, segment in enumerate(trn_loss_segments):
                plt.plot(segment, label=f'Train Loss Fold {i + 1}')
            for i, segment in enumerate(val_loss_segments):
                plt.plot(segment, label=f'Validation Loss Fold {i + 1}', linestyle='--')
            plt.title('Training Progress')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

            plt.tight_layout()

            file_name = f"{config_name}_history_{now}.png"
            plt.savefig(f"{output_dir}/{file_name}")
            # plt.show()

            df_metrics = pd.DataFrame(history)
            df_metrics = df_metrics.iloc[epochs-1::epochs] #fold 단위로만 기록. 다남기고 싶으면 주석처리 or 파라미터화
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


    def test(self, model_path: str):
        if len(self.configs) > 1:
            raise ValueError("config name must be specified")
        config = self.configs[0]

        model = config['model']
        model_class = model['class']
        module_list = model['module_list']
        
        data = config['data']

        test_X_df = pd.read_csv(data['test_X']['path'], index_col=data['test_X']['index_col'])
        test_y_df = pd.read_csv(data['test_y']['path'], index_col=data['test_y']['index_col'])

        transform = data['transform'] if 'transform' in data else None
        if transform is not None:
            transform_steps = transform['steps']
            for step in transform_steps:
                for transform_obj, params in step.items():
                    obj_instance = transform_obj(**params["params"])
                    if obj_instance.__class__ == OneHotEncoder:
                        targets = params["fit_transform_cols"]
                        # 원-핫 인코딩 적용할 컬럼 선택
                        onehot_encoded = obj_instance.fit_transform(test_X_df[targets])
                        # 원-핫 인코딩된 데이터프레임 생성
                        onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=obj_instance.get_feature_names_out(targets))
                        # 원래 데이터프레임에서 인코딩 대상 컬럼 제거
                        test_X_df = test_X_df.drop(targets, axis=1)
                        # 원-핫 인코딩된 데이터프레임과 원래 데이터프레임 결합
                        test_X_df = pd.concat([test_X_df, onehot_encoded_df], axis=1)
                    elif hasattr(obj_instance, 'fit_resample'):
                        targets = params["fit_resample_cols"]
                        test_X_df, test_y_df = getattr(obj_instance, 'fit_resample')(test_X_df, test_y_df)
                    elif hasattr(obj_instance, 'fit_transform'):
                        targets = params["fit_transform_cols"]
                        test_X_df[targets] = getattr(obj_instance, 'fit_transform')(test_X_df[targets])
                    else:
                        raise ValueError("transform object must have fit_transform or fit_resample method")
        print(f"transformed test_X_df.shape: {test_X_df.shape}")
        print(f"transformed test_y_df.shape: {test_y_df.shape}")

        X = torch.Tensor(test_X_df.to_numpy(dtype=np.float32))
        y = torch.Tensor(test_y_df.to_numpy(dtype=np.float32))

        #테스트셋의 input 개수로 모델의 input 개수 조정
        module_list[0] = nn.Linear(X.shape[1], module_list[0].out_features)

        device = config['hyper_params']['device']
        print(f"running device: {device}")

        model: nn.Module = model_class(module_list).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.inference_mode():
            y_hat = model.forward(X)

            preds = y_hat > 0.5
            preds_int = preds.to(torch.float32) 

            accuracy = (preds_int == y).sum().float() / len(y)
            print(f"Accuracy: {accuracy:.4f}")
      

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