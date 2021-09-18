from lightgbm import LGBMRegressor

default_params = {

}

search_params = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'learning_rate': {'min': 0.0001, 'max': 0.5},
    'num_leaves': {'min': 10, 'max':100},
    'n_estimators': {'min': 5, 'max': 500}
}

class LGBM:
    def __init__(self):
        self._model = None
        self._params = default_params
        self._search_params = search_params
    
    def create(self, params=default_params):
        self._params = params
        self._model = LGBMRegressor(**self._params)
    
    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)
    
    def predict(self, X):
        return self._model.predict(X)