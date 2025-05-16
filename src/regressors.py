#@article{scikit-learn,
#  title={Scikit-learn: Machine Learning in {P}ython},
#  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
#          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
#          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
#          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
#  journal={Journal of Machine Learning Research},
#  volume={12},
#  pages={2825--2830},
#  year={2011}
#}

import sys
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import xgboost as xgb

import numpy as np

import operations as op
import regressor_test as test


VERBOSE = False

REGRESSOR_TABLE = {}

RANDOM_STATE = 0

def set_random_state(new_random_state):
    global RANDOM_STATE

    RANDOM_STATE = new_random_state

class BaseRegressor:

    def __init__(self, **params):
        self.set_params(**params)

    def refresh(self, **params):
        self.set_params(**self.get_params())
        return self

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params): 
        self.params = params
        # print(f'*** {type(self)}:create:{self.params}')
        self._regressor = self.create_regressor(**params)
        return self

    def is_regressor(self):
        return True

    def create_regressor(self, **params):
        raise Exception("INCOMPLETE")

    def hyperpar_space(self):
        raise Exception(f'hyperpar_space() not defind for class {type(self)}')

    def param_grid(self):
        raise Exception("INCOMPLETE")

    def fit(self, X_train, y_train):
        if VERBOSE:
            print(f'- Regressor: X_TRAIN:\n{X_train.to_string()}\n')
            print(f'- Regressor: y_train:\n{y_train.to_string()}\n')
        if True:
            self.X_train = X_train
            self.y_train = y_train
        self._fitted = self._regressor.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        if hasattr(self, '_fitted'):
            res =  self._fitted.predict(X_test)
        else:
            raise Exception("UNEXPECTED")
        return res

   

    def y_neg(self, predictions):
        return len(list(filter(lambda x: (x < 0), predictions)))

    def y_rmse(self, y_true, predictions):
        if VERBOSE:
            df = pd.concat([pd.Series(y_true),pd.Series(predictions)],axis=1,keys=["y_true","predict"])
            print(f'{df.to_string()}')
            print(f'RMSE={mean_squared_error(y_true, predictions, squared = False)}')
        return mean_squared_error(y_true, predictions, squared = False)

    def y_normrmse(self, y_true, predictions):
        rmse = mean_squared_error(y_true, predictions, squared=False)
        normalized_rmse = rmse
        
        # Calculate the range of the true values
        y_range = np.max(y_true) - np.min(y_true)
        # Calculate the normalized RMSE
        normalized_rmse = rmse / y_range
        

        
        
        
        if VERBOSE:
            print(f'Normalized RMSE={normalized_rmse}')
        
        return normalized_rmse
    
    
    

    def y_mse(self, y_true, predictions):
        return mean_squared_error(y_true, predictions, squared = True)

    def y_mae(self, y_true, predictions):
        y_true, predictions = np.array(y_true), np.array(predictions)
        return np.mean(np.abs(y_true - predictions)) 

    def y_r2_score(self, y_true, predictions):
        return r2_score(y_true, predictions)

    

    def predict_score(self, X_test, y_test, d_score, pfx='y'):
        op.TIMER.start()
        if False and pfx == 'y_test':
            test.save_Xy(self.X_train, self.y_train, X_test, y_test)
            # y_predict = test.plain_test(verbose=True, trained_regressor=self._fitted)
        y_predict = self.predict(X_test)
        op.TIMER.stop()
        d_score[f"{pfx}_rmse"] = self.y_rmse(y_test.values, y_predict)
        d_score[f"{pfx}_mse"] = self.y_mse(y_test.values, y_predict)
        d_score[f"{pfx}_mae"] = self.y_mae(y_test.values, y_predict)
        d_score[f"{pfx}_r2"] = self.y_r2_score(y_test.values, y_predict)
        d_score[f"{pfx}_normrmse"] = self.y_normrmse(y_test.values, y_predict)
        #
     
        #
        if VERBOSE:
            d_score[f"{pfx}_neg"] = self.y_neg(y_predict)
            d_score[f"{pfx}_neg_true"] = self.y_neg(y_test)
        d_score[f"t_{pfx}_predict"] = op.TIMER.last_seconds()
        return y_predict

#
#
#

class my_MLPRegressor(BaseRegressor):

    def create_regressor(self, **params):
        return MLPRegressor(**params) 

    




REGRESSOR_TABLE['mlpregressor'] = my_MLPRegressor(
                                **{
                                    "random_state":1,
                                    # "learning_rate_init":0.01,
                                    "learning_rate_init":0.001,
                                    "max_iter":1000
                                })



class my_AdaBoostRegressor(BaseRegressor):

    def create_regressor(self, **params):
        return AdaBoostRegressor(**params)

    
REGRESSOR_TABLE['adaboostregressor'] = my_AdaBoostRegressor(
                                **{
                                    "random_state":0,
                                    "n_estimators":100
                                })





class my_ridgeregression(BaseRegressor):

    def create_regressor(self, **params):
        return Ridge(**params)



REGRESSOR_TABLE['ridgeregression'] = my_ridgeregression(
                                **{
                                })





class my_svregression(BaseRegressor):

    def create_regressor(self, **params):
        return SVR(**params)


REGRESSOR_TABLE['svregression'] = my_svregression(
                                **{
                                    # empty
                                })


class my_knn(BaseRegressor):

    def create_regressor(self, **params):
        return KNeighborsRegressor(**params)
    

REGRESSOR_TABLE['knn'] = my_knn(
                                **{
                                    # empty
                                })
#
#
#

class my_xgboost(BaseRegressor):

    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self._model = xgb.train(self.get_params(), dtrain)#seed???

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        predictions = self._model.predict(dtest)
        return predictions

    def coeff(self, X_test, y_true):
        return -1 # INCOMPLETE

    def create_regressor(self, **params):
        # None is returned because the object self is the regressor
        return None

   
REGRESSOR_TABLE['xgboost'] = my_xgboost(
                                **{
                                    'objective': 'reg:squarederror',
                                    'max_depth': 3, #6
                                    'learning_rate': 0.3,
                                    'seed':0
                                })



class my_lightgbm(BaseRegressor):

    def fit(self, X_train, y_train):
        lgb_train = lgb.Dataset(X_train, y_train)
   
    
        self._model = lgb.train(self.get_params(),
                        train_set=lgb_train
                      
                        )

    def predict(self, X_test):
        predictions = self._model.predict(X_test)
        return predictions

    def coeff(self, X_test, y_true):
        return -1 # INCOMPLETE

    def create_regressor(self, **params):
        # None is returned because the object self is the regressor
        return None
    
REGRESSOR_TABLE['lightgbm'] = my_lightgbm(
                                **{
                                    'task': 'train',
                                    'boosting': 'gbdt',
                                    'objective': 'regression',
                                    'num_leaves': 31,
                                    'verbose': 1,
                                    'seed':0
                                })



def get_regressor(p_name):
    name = p_name.lower()
    if name in REGRESSOR_TABLE:
        return REGRESSOR_TABLE[name].refresh()
    else:
        raise Exception(f'unknown regressor "{p_name}" ')
