from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint

config = {
    'files': {
    'X_csv': '/home/sangjun/work/temp/trn_X_20231211_final_col_01_transformed.csv',
    'y_csv': '/home/sangjun/work/temp/trn_y_20231211_final_col_01_transformed.csv',
    'output': './model.pth',
    'output_csv': './results/cv.csv',
  	},
      
    'model': RandomForestClassifier,
    
    'model_params': {
		'n_estimators': randint(10, 100),
		'max_depth': randint(1, 10),
		'min_samples_split': randint(2, 20),
		'min_samples_leaf': randint(1, 20)
  	},

	'random_params': {
        'n_iter':10,
        'cv':5,
        'scoring':'accuracy'
	},
    
	'cvld_params': {
        'scoring' : ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'cv':StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
	}
}