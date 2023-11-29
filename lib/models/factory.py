import importlib
import torch.nn as nn

class ModelBuilder():
    
    @staticmethod
    def make(model: str, params: dict) -> nn.Module:
        match model.lower():
            case "ann":
                from lib.models.ann import ANN

                model_config_file = params.model_config_file
                package = model_config_file.split(".")[0]
                package = package.replace("/", ".")

                package = importlib.import_module(package)

                hyper_params = getattr(package, 'HyperParams')
                model_params = getattr(package, 'ModelParams')

                return ANN(
                    perceptron=model_params.perceptron,
                    optim=model_params.optim,
                    loss_function=model_params.loss_function,
                    dropout=hyper_params.DROP_OUT,
                    lr=hyper_params.LEARNING_RATE,
                    device=hyper_params.DEVICE,
                ).to(hyper_params.DEVICE)
            case _:
                raise ValueError("model must be needed")