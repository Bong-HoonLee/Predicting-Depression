import pandas as pd
import numpy as np

import os

root_dir = os.getcwd()

train_df = pd.read_csv(f"{root_dir}/data/train.csv")
train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})

test_df = pd.read_csv(f"{root_dir}/data/test.csv")
test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})

X_train = pd.get_dummies(train_df[["Pclass", "Sex", "SibSp", "Parch"]])
y_train = train_df[["Survived"]]

X_test = test_df[["Pclass", "Sex", "SibSp", "Parch"]]

X_train.to_csv(f"{root_dir}/data/train_X.csv", index=False)
y_train.to_csv(f"{root_dir}/data/train_y.csv", index=False)

X_test.to_csv(f"{root_dir}/data/test_X.csv", index=False)
