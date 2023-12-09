import torch
import torch.nn as nn
import torchmetrics

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from models.ann import ANN

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

#eda01 (from archive/corr_heatmap_test.ipynb)
numerical = ['age', 'HE_wc', 'ainc', 'HE_ht', 'HE_wt']
onehot = ['LQ1_sb', 'BO1', 'DI1_pt', 'DE1_pr', 'DI2_pr', 'BH2_61', 'DI3_pr', 'BO1_1', 'marri_1', 'DI2_pt', 'BD1', 'EC_stt_2', 'DI3_pt', 'live_t', 'npins', 'EC_wht_0', 'LQ2_ab', 'BH1', 'D_2_1', 'DJ4_pr', 'DJ4_pt', 'DI1_pr', 'LQ4_00', 'DE1_pt', 'educ', 'EC_occp', 'sex']
label = ['BE5_1', 'BE3_31', 'D_1_1', 'cfam', 'ho_incm', 'BP1', 'incm5', 'edu', 'incm', 'DI3_2', 'ho_incm5']

y_related = ['BP_PHQ_1', 'BP_PHQ_2', 'BP_PHQ_3', 'BP_PHQ_4', 'BP_PHQ_5', 'BP_PHQ_6', 'BP_PHQ_7', 'BP_PHQ_8', 'BP_PHQ_9', 'mh_PHQ_S', 'BP6_10', 'BP6_31', 'DF2_pr', 'DF2_pt', 'BP1']
y = "depressed"

config = {
    "name": "train_X_eda01",
    "model": {
        "class": ANN,
        "module_list": nn.ModuleList([
            nn.Linear(111, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ]),
    },
    "data": {
        "train_X": {
            "path": "data/train_X_eda01.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "data/train_y_eda01.csv",
            "index_col": None,
        },
        "test_X": {
            "path": "data/HN_X_231206_wo_y_test.csv",
            "index_col": None,
        },
        "test_y": {
            "path": "data/HN_y_231206_wo_y_test.csv",
            "index_col": None,
        },
        "transform": {
            "steps": [
                {
                    KNNImputer: {
                        "params": {
                            "n_neighbors": 5,
                            "weights": "uniform",
                            "missing_values": float("nan"),
                        },
                        "fit_transform_cols": numerical
                    }
                },
                {
                    SimpleImputer: {
                        "params": {
                            "missing_values": float("nan"),
                            "strategy": "most_frequent",
                        },
                        "fit_transform_cols": onehot
                    }
                },
                {
                    SimpleImputer: {
                        "params": {
                            "strategy": "most_frequent",
                        },
                        "fit_transform_cols": label
                    }
                },
                {
                    MinMaxScaler: {
                        "params": {
                            "feature_range": (0, 1),
                        },
                        "fit_transform_cols": numerical
                    }
                },
                {
                    OneHotEncoder: {
                        "params": {
                            "sparse": False,
                        },
                        "fit_transform_cols": onehot
                    }
                },
                {
                    RandomUnderSampler: {
                        "params": {
                            "sampling_strategy": 0.1,
                            "random_state": 42,
                        },
                    }
                },
                {
                    SMOTE: {
                        "params": {
                            "random_state": 42,
                        },
                    }
                }
            ],
        },
        "output_dir": "output/",
    },
    "hyper_params": {
        "data_loader_params": {
            "batch_size": 32,
            "shuffle": True,
        },
        "loss": nn.BCELoss(),
        "optim": torch.optim.Adam,
        "optim_params": {
            "lr": 0.00001,
        },
        "metrics": torchmetrics.MetricCollection({
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'f1score': BinaryF1Score(),
            'auroc': BinaryAUROC(),
        }),
        "device": "cuda"
        if torch.cuda.is_available()
        else "cpu",
        "epochs": 100,
        'cv_params':{
            'n_split': 5,
        },
    },
}
