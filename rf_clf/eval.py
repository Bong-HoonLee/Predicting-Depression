from sklearn.model_selection import cross_validate, RandomizedSearchCV
import pandas as pd

# RandomSearch
def rs(X, y, model, param_dist, n_iter, cv, scoring):
    model = model()
    param_dist = param_dist

    random_search = RandomizedSearchCV(model,
                                        param_distributions=param_dist,
                                        n_iter=n_iter,
                                        cv=cv,
                                        scoring = scoring
                                        )
    random_search.fit(X, y)

    return random_search.best_params_

# cv
def cvld(X, y, model, n_estimators, max_depth, min_samples_leaf, min_samples_split, scoring, cv):
    scoring = scoring

    model = model(n_estimators = n_estimators, 
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split)
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv)

    scores_df = pd.DataFrame(scores)
    
    return pd.concat([scores_df, scores_df.apply(['mean', 'std'])])

def get_args_parser(add_help=True):
    import argparse
  
    parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    import numpy as np

    args = get_args_parser().parse_args()

    exec(open(args.config).read())
    cfg = config

    # get X, y (전처리 후 shape, ndim 확인 필수)
    files = cfg.get('files')
    X_df = pd.read_csv(files.get('X_csv'))
    y_df = pd.read_csv(files.get('y_csv'))
    X, y = X_df.to_numpy(dtype=np.float32), y_df.to_numpy(dtype=np.float32).flatten()
    
    # model
    model = cfg.get('model')
    # randomized search setting
    model_params = cfg.get('model_params')
    random_params = cfg.get('random_params')
    n_iter = random_params.get('n_iter')
    cv = random_params.get('cv')
    scoring = random_params.get('scoring')
    # randomized search
    bestParameter = rs(X, y, model, model_params, n_iter, cv, scoring)
    print(bestParameter)
    # model prams
    max_depth = bestParameter['max_depth']
    min_samples_leaf = bestParameter['min_samples_leaf']
    min_samples_split = bestParameter['min_samples_split']
    n_estimators = bestParameter['n_estimators']
    # cv setting
    cvld_params = cfg.get('cvld_params')
    scoring = cvld_params.get('scoring')
    cv = cvld_params.get('cv')
    print(cv)
    # cv
    cv_scores = cvld(X, y, model, n_estimators, max_depth, min_samples_leaf, min_samples_split, scoring, cv)
    print(cv_scores)

    cv_scores.to_csv(files.get('output_csv'))
    