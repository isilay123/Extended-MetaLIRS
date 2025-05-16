import os
import pandas as pd
from sklearn.metrics import mean_squared_error 
import operations as op

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge

from impute_utils import generate_html

import pandas as pd

VERBOSE = True

X_TRAIN_FILE = f'{os.path.join("REGRESSOR_TEST","X_train.csv")}'
X_TEST_FILE = f'{os.path.join("REGRESSOR_TEST","X_test.csv")}'
Y_TRAIN_FILE = f'{os.path.join("REGRESSOR_TEST","y_train.csv")}'
Y_TEST_FILE = f'{os.path.join("REGRESSOR_TEST","y_test.csv")}'
Y_TEST_PREDICT_FILE = f'{os.path.join("REGRESSOR_TEST","y_test_predict.csv")}'

def save_Xy(X_train,y_train,X_test,y_test):
    print("****** PREPARING REGRESSOR_TEST *********")
    X_train.to_csv(X_TRAIN_FILE, index=False)
    y_train.to_csv(Y_TRAIN_FILE, index=False)
    X_test.to_csv(X_TEST_FILE, index=False)
    y_test.to_csv(Y_TEST_FILE, index=False)

def load_Xy():
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_train.reset_index(drop=True,inplace=True)
    X_test = pd.read_csv(X_TEST_FILE)
    X_test.reset_index(drop=True,inplace=True)
    y_train = pd.read_csv(Y_TRAIN_FILE)
    y_train.reset_index(drop=True,inplace=True)
    y_test = pd.read_csv(Y_TEST_FILE)
    y_test.reset_index(drop=True,inplace=True)
    if True:
        print('*** load_Xy():')
        print(f'X_train =\n{X_train}')
        print(f'y_train =\n{y_train}')
        print(f'X_test =\n{X_test}')
        print(f'y_test =\n{y_test}')
    return (X_train,y_train,X_test,y_test)

def plain_test(verbose=VERBOSE):
    X_train,y_train,X_test,y_test = load_Xy()
    regressor = LinearRegression()
    # regressor = AdaBoostRegressor()
    # regressor = Ridge()
    trained_regressor = regressor.fit(X_train, y_train)
    # print(f'*** _COEF = {trained_regressor.coef_}')
    y_predict = trained_regressor.predict(X_test)
    if verbose:
        rmse = mean_squared_error(y_test, y_predict, squared = False)
        print(f'*** PLAIN TEST: rmse = {rmse}')
    if True:
        generate_html(f'{os.path.join("REGRESSOR_TEST","X_train.html")}', X_train, X_train, X_train, y_predict=y_train, y_original=y_train, label="X_train")
        generate_html(f'{os.path.join("REGRESSOR_TEST","X_test.html")}', X_test, X_test, X_test, y_predict=y_predict, y_original=y_test, label="X_test")
    pd.DataFrame(y_predict).to_csv(Y_TEST_PREDICT_FILE, index=False)
    return y_predict
    
if __name__ == "__main__":
    y_predict = plain_test()
