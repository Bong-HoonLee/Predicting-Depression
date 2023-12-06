import torch
import torch.nn as nn
import torchmetrics
import pandas as pd

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from models.ann import ANN

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

numerical = ['age', 'ainc', 'HE_sbp', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_glu', 'HE_chol', 'HE_HDL_st2', 'HE_TG', 'HE_ast', 'HE_alt', 'HE_HB', 'HE_HCT', 'HE_BUN', 'HE_crea', 'HE_WBC', 'HE_RBC', 'HE_Bplt', 'HE_Uph', 'HE_Usg']
onehot = ['sex', 'occp', 'marri_1', 'tins', 'npins', 'D_2_1', 'DI1_pr', 'DI1_pt', 'DI2_pr', 'DI2_pt', 'DI3_pr', 'DI3_pt', 'DJ4_pr', 'DJ4_pt', 'DE1_pr', 'DE1_pt', 'BH9_11', 'BH1', 'BH2_61', 'LQ4_00', 'LQ1_sb', 'LQ2_ab', 'EC_occp', 'EC_stt_1', 'EC_stt_2', 'BO1_1', 'BO2_1', 'BD1', 'BS8_2', 'BS9_2', 'HE_rPLS', 'HE_Unitr', 'HE_Upro', 'HE_Uglu', 'HE_Uket', 'HE_Ubil', 'HE_Ubld', 'HE_Uro', 'BM1_1', 'BM1_2', 'BM1_3', 'BM1_4', 'BM1_5', 'BM1_6', 'BM1_7', 'BM1_8', 'live_t', 'educ', 'BO1', 'HE_obe']
label = ['incm', 'ho_incm', 'incm5', 'ho_incm5', 'edu', 'cfam', 'house', 'D_1_1', 'DI3_2', 'BD1_11', 'BD2_1', 'BA2_12', 'BA2_13', 'BP1', 'BS3_1', 'BE3_31', 'BE5_1']
y_related = ["BP_PHQ_1", "BP_PHQ_2", "BP_PHQ_3", "BP_PHQ_4", "BP_PHQ_5", "BP_PHQ_6", "BP_PHQ_7", "BP_PHQ_8", "BP_PHQ_9", "mh_PHQ_S", "BP6_10", "BP6_31", "DF2_pr", "DF2_pt"]
y = "depressed"

config = {
    "name": "HN_X_231206",
    "model": {
        "class": ANN,
        "module_list": nn.ModuleList([
            nn.Linear(224, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ]),
    },
    "data": {
        "train_X": {
            "path": "data/HN_X_231206.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "data/HN_y_231206.csv",
            "index_col": None,
        },
        "test_X": {
            "path": "data/HN_test_X.csv",
            "index_col": None,
        },
        "test_y": {
            "path": "data/HN_test_y.csv",
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
                        "params": {},
                        "fit_transform_cols": onehot
                    }
                },
                {
                    RandomUnderSampler: {
                        "params": {
                            "sampling_strategy": 0.1,
                            "random_state": 42,
                        },
                        "fit_resample_cols": y_related + label
                    }
                },
                {
                    SMOTE: {
                        "params": {
                            "random_state": 42,
                        },
                        "fit_resample_cols": y_related + label
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
            "lr": 0.01,
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
