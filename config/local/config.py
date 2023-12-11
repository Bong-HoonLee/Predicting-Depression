import torch
import torch.nn as nn
import torchmetrics

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from models.ann import ANN

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from torch.optim.lr_scheduler import ReduceLROnPlateau

#eda01 (from archive/corr_heatmap_test.ipynb)
numerical = ['HE_chol', 'HE_Bplt', 'HE_wt', 'HE_ht', 'HE_RBC', 'HE_HDL_st2', 'HE_HB', 'age', 'HE_Uph', 'HE_crea', 'HE_Usg', 'HE_WBC', 'HE_TG', 'HE_alt', 'HE_ast',
 'HE_HCT', 'HE_dbp', 'HE_glu', 'HE_BUN', 'ainc', 'HE_sbp', 'HE_wc']
onehot = ['BO2_1', 'educ', 'BH1', 'EC_stt_2', 'sex', 'HE_Unitr', 'LQ4_00', 'DI1_pt', 'BM1_5', 'DE1_pt', 'DJ4_pt', 'HE_Ubld', 'LQ1_sb', 'BM1_8', 'occp', 'DI2_pr',
 'BD1', 'marri_1', 'DI3_pr', 'BM1_3', 'EC_occp', 'BM1_4', 'live_t', 'HE_Ubil', 'BH2_61', 'DI1_pr', 'EC_stt_1', 'BM1_7', 'DJ4_pr', 'DE1_pr', 'HE_Uglu',
 'HE_Uro', 'HE_Upro', 'HE_Uket', 'HE_obe', 'HE_rPLS', 'BO1_1', 'BO1', 'D_2_1', 'BM1_1', 'npins', 'LQ2_ab', 'BH9_11', 'tins', 'DI2_pt', 'BM1_2',
 'DI3_pt', 'BS8_2', 'BS9_2', 'BM1_6']
label = ['BD1_11', 'incm', 'ho_incm', 'incm5', 'ho_incm5', 'D_1_1', 'cfam', 'BA2_13', 'BS3_1', 'DI3_2', 'house', 'BA2_12', 'BE3_31', 'edu', 'BD2_1', 'BE5_1']
y_related =["BP_PHQ_1","BP_PHQ_2","BP_PHQ_3","BP_PHQ_4","BP_PHQ_5","BP_PHQ_6","BP_PHQ_7","BP_PHQ_8","BP_PHQ_9","mh_PHQ_S","BP6_10","BP6_31","DF2_pr","DF2_pt","BP1"]
y = "depressed"

config = {
    "name": "train_X_231211",
    "model": {
        "class": ANN,
        "module_list": nn.ModuleList([
            nn.Linear(220, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ]),
    },
    "data": {
        "train_X": {
            "path": "localdata/train_X_231211.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "localdata/train_y_231211.csv",
            "index_col": None,
        },
        "test_X": {
            "path": "localdata/test_X_231211.csv",
            "index_col": None,
        },
        "test_y": {
            "path": "localdata/test_y_231211.csv",
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
                            "sampling_strategy": 1,
                            "random_state": 42,
                        },
                    }
                },
                {
                    ADASYN: {
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
            "batch_size": 64,
            "shuffle": True,
        },
        "loss": nn.BCELoss(),
        "optim": torch.optim.AdamW,
        "optim_params": {
            "lr": 0.0001,
            'weight_decay': 0,
        },
        "metrics": torchmetrics.MetricCollection({
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'f1score': BinaryF1Score(),
            'auroc': BinaryAUROC(),
        }),
        "device": "cpu"
        if torch.cuda.is_available()
        else "cpu",
        "epochs": 100,
        'cv_params':{
            'n_split': 5,
        },
        'lr_scheduler': ReduceLROnPlateau,
        'scheduler_params': {
            'mode': 'min',
            'factor': 0.1,
            'patience': 5,
            'verbose':False
    },
    },
}
