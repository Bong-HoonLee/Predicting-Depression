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


numerical = ['HE_chol', 'HE_HCT', 'HE_ast', 'HE_sbp', 'HE_Uph', 'age', 'HE_TG', 'HE_wc', 'HE_HDL_st2', 'HE_WBC', 'HE_glu', 'HE_crea', 'HE_BUN', 'HE_Bplt', 'HE_wt', 'HE_HB', 'ainc', 'HE_RBC', 'HE_Usg', 'HE_ht', 'HE_HbA1c', 'HE_dbp', 'HE_alt']
onehot = ['BO1_1', 'DI2_pr', 'BH9_11', 'BM1_0', 'EC_stt_2', 'HE_Uglu', 'live_t', 'educ', 'BM1_4', 'HE_Ubld', 'HE_Ubil', 'occp', 'BH1', 'LQ2_ab', 'npins', 'marri_1', 'D_2_1', 'DE1_pr', 'HE_Unitr', 'sex', 'HE_Uket', 'LK_LB_EF', 'BS8_2', 'DE1_pt', 'HE_Upro', 'BM1_7', 'DJ4_pt', 'DI3_pt', 'BH2_61', 'EC_occp', 'BS9_2', 'BO1', 'DJ4_pr', 'BD1', 'BM1_8', 'BM1_3', 'LQ1_sb', 'BM1_1', 'DI1_pr', 'BM1_2', 'HE_rPLS', 'HE_obe', 'region', 'DI2_pt', 'DI3_pr', 'tins', 'EC_stt_1', 'LQ4_00', 'BM1_6', 'BO2_1', 'HE_Uro', 'BM1_5', 'DI1_pt']
label = ['incm5', 'DI3_2', 'cfam', 'BD2_1', 'ho_incm5', 'edu', 'house', 'BA2_12', 'ho_incm', 'BE3_31', 'BE5_1', 'OR1', 'BS3_1', 'D_1_1', 'incm', 'BD1_11', 'BA2_13']
y_related = ['BP_PHQ_1', 'BP_PHQ_2', 'BP_PHQ_3', 'BP_PHQ_4', 'BP_PHQ_5', 'BP_PHQ_6', 'BP_PHQ_7', 'BP_PHQ_8', 'BP_PHQ_9', 'mh_PHQ_S', 'BP6_10', 'BP6_31', 'DF2_pr', 'DF2_pt', 'BP1']



config = {
    "name": "train_X_231211_final_col_03",
    "model": {
        "class": ANN,
        "module_list": nn.ModuleList([
            nn.Linear(243, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ]),
    },
    "data": {
        "train_X": {
            "path": "data/trn_X_20231211_final_col_03.csv",
            "index_col": None,
        },
        "train_y": {
            "path": "data/trn_y_20231211_final_col_03.csv",
            "index_col": None,
        },
        "test_X": {
            "path": "data/tst_X_20231211_final_col_03.csv",
            "index_col": None,
        },
        "test_y": {
            "path": "data/tst_y_20231211_final_col_03.csv",
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
            "batch_size": 128,
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
        "epochs": 10,
        'cv_params':{
            'n_split': 5,
        },
    },
}
