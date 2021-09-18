from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import numpy as np

import model_lightgbm
import optuna

import global_valiables

_modeler = None
_last_params = None
_X_train = None
_y_train = None
_X_test = None

def setup(modeltype, X_train, y_train, X_test):
    global _modeler, _last_params, _X_train, _y_train, _X_test

    if modeltype=='lightgbm':
        _modeler = model_lightgbm.LGBM()
    else:
        raise Exception('modeltype: {} is not defined'.format(modeltype))
    
    _modeler.create()
    _last_params = _modeler._params
    _X_train = X_train
    _y_train = y_train
    _X_test = X_test

def score_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def objective(trial):
    for key, val in _modeler._search_params.items():
        if type(val) == list:
            _last_params[key] = trial.suggest_categorical(key, val)
        elif type(val) == dict:
            if type(val['min']) == float:
                _last_params[key] = trial.suggest_float(key, val['min'], val['max'])
            elif type(val['min']) == int:
                _last_params[key] = trial.suggest_int(key, val['min'], val['max'])
    
    _modeler.create(_last_params)
    score_funcs = {
        'rmse': make_scorer(score_rmse)
    }
    scores = cross_validate(_modeler._model, _X_train, _y_train, cv=5, scoring=score_funcs)

    return scores['test_rmse'].mean()


def experiment(n_trials):
    global _modeler, _last_params, _X_train, _y_train, _X_test

    if n_trials==0:
        _modeler.create()
    else:
        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
    _modeler.fit(_X_train, _y_train)


def predict():
    global _modeler, _last_params, _X_train, _y_train, _X_test

    train_score = np.sqrt(mean_squared_error(_y_train, _modeler.predict(_X_train)))
    print('train RMSE: {}'. format(train_score))

    return _modeler.predict(_X_test)

    