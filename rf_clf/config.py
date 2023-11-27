from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

config = {
    'files': {
    'X_csv': './trn_X.csv',
    'y_csv': './trn_y.csv',
    'output': './model.pth',
    'output_csv': './results/cv.csv',
  	},
      
    'model': RandomForestClassifier,
    
    'model_params': {
		'n_estimators': randint(10, 300),
		'max_depth': randint(1, 20),
		'min_samples_split': randint(2, 20),
		'min_samples_leaf': randint(1, 20)
  	},

	'random_params': {
        'n_iter':10,
        'cv':5,
        'scoring':'accuracy'
	},
    
	'cvld_params': {
        'scoring' : ['accuracy', 'precision', 'recall', 'f1'],
        'cv':5
	}
}