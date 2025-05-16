from .imputer_base import ImputerBase
from .mlp_imputer import MLPImputer

import pandas as pd
import numpy as np

IMPUTE_NAME = "MLP2"


#
# MLP2 imputer, documentation can be found at
#


TRAIN_ALL = True # Use all data for training


class Imputer(ImputerBase):

    def __init__(self):
        super().__init__(IMPUTE_NAME)
        self.mlpi = MLPImputer(train_all=TRAIN_ALL, verbose=False)

    def impute(self, tensor_X, df_train=None, column_names=None):
        if df_train is None:
            X = pd.DataFrame(tensor_X.numpy().astype(np.float64))
            if column_names is not None:
                X.columns = column_names
            mlpi = MLPImputer(train_all=TRAIN_ALL, verbose=False)
            mlpi.fit(X)
            imputed_X = mlpi.transform(X)
        else:
            X = pd.DataFrame(df_train.numpy().astype(np.float64))
            if column_names is not None:
                X.columns = column_names
            mlpi = MLPImputer(train_all=TRAIN_ALL, verbose=False)
            mlpi.fit(X)
            #
            X_trf = pd.DataFrame(tensor_X.numpy().astype(np.float64))
            if column_names is not None:
                X_trf.columns = column_names
            imputed_X = mlpi.transform(X_trf)
        return imputed_X.to_numpy()



if __name__ == "__main__":
   Imputer().random_test()

