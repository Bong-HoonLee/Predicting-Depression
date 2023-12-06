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

numerical = ['HE_RBC', 'HE_glu', 'HE_wt', 'HE_TG', 'HE_ast', 'HE_crea', 'ainc', 'HE_ht', 'HE_chol', 'HE_Uph', 'HE_Bplt', 'HE_wc', 'HE_dbp', 'HE_Usg', 'HE_HDL_st2', 'HE_WBC', 'HE_HCT', 'HE_HB', 'age', 'HE_alt', 'HE_sbp', 'HE_BUN']
onehot = ['DI2_pt', 'HE_Uglu', 'BS8_2', 'sex', 'BO1_1', 'HE_Unitr', 'occp', 'DI1_pr', 'HE_Upro', 'BM1_1', 'DI3_pr', 'HE_Ubld', 'BO2_1', 'BO1', 'DI1_pt', 'BM1_3', 'BM1_6', 'HE_obe', 'tins', 'BM1_4', 'marri_1', 'BH9_11', 'HE_Uket', 'DE1_pr', 'BH2_61', 'BH1', 'D_2_1', 'DE1_pt', 'HE_rPLS', 'HE_Ubil', 'live_t', 'BM1_5', 'BM1_8', 'DI2_pr', 'HE_Uro', 'BD1', 'DI3_pt', 'BM1_2', 'BS9_2', 'BM1_7', 'DJ4_pt', 'LQ4_00', 'npins', 'educ', 'EC_occp', 'EC_stt_2', 'LQ1_sb', 'EC_stt_1', 'DJ4_pr', 'LQ2_ab']
label = ['cfam', 'BE5_1', 'house', 'BA2_13', 'BS3_1', 'D_1_1', 'incm', 'ho_incm', 'incm5', 'BE3_31', 'BA2_12', 'edu', 'BD2_1', 'DI3_2', 'ho_incm5', 'BD1_11']
y_related = ['BP_PHQ_1', 'BP_PHQ_2', 'BP_PHQ_3', 'BP_PHQ_4', 'BP_PHQ_5', 'BP_PHQ_6', 'BP_PHQ_7', 'BP_PHQ_8', 'BP_PHQ_9', 'mh_PHQ_S', 'BP6_10', 'BP6_31', 'DF2_pr', 'DF2_pt', 'BP1']
y = "depressed"

config = {
    "name": "HN_X_231207_only_numerical",
    "model": {
        "class": ANN,
        "module_list": nn.ModuleList([
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ]),
    },
    "data": {
        "train_X": {
            "path": "data/HN_X_numerical_231206_wo_transform.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "data/HN_y_numerical_231206.csv",
            "index_col": None,
        },
        "test_X": {
            "path": "data/HN_X_231206_numerical_wo_transform_test.csv",
            "index_col": None,
        },
        "test_y": {
            "path": "data/HN_y_231206_numerical__test.csv",
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
                    MinMaxScaler: {
                        "params": {
                            "feature_range": (0, 1),
                        },
                        "fit_transform_cols": numerical
                    }
                },
                {
                    RandomUnderSampler: {
                        "params": {
                            "sampling_strategy": 0.5,
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
            "lr": 0.0001,
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
        "epochs": 50,
        'cv_params':{
            'n_split': 5,
        },
    },
}
