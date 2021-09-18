from sklearn.ensemble import RandomForestRegressor

default_params = {
    'random_state': 123
}

search_params = {
    'n_estimators': {'min':10, 'max': 2000},
    'max_features': ['auto', 'sqrt', 'log2']
}

class RandomForest:
    def __init__(self):
        self._model = None
        self._params = default_params
        self._search_params = search_params
    
    def create(self, params=default_params):
        self._params = params
        self._model = RandomForestRegressor(**self._params)
    
    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X):
        return self._model.predict(X)