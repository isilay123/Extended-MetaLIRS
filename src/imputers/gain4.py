import sys
from .imputer_base import ImputerBase
from ._jager_generative_imp import GAINImputer

import numpy as np
import pandas as pd


IMPUTE_NAME = "GAIN4"


# We utilized that code: https://github.com/se-jaeger/data-imputation-paper/blob/main/src/data_imputation_paper/imputation/generative.py


EXCEPTION_RETRY = True


class Imputer(ImputerBase):

    def __init__(self):
        super().__init__(IMPUTE_NAME)

    def _run_impute(self, df, column_names=None, df_train=None):
        #
        if df_train is None:
            df_train = df
        #
        X_missing = pd.DataFrame(df_train.numpy())
        X_missing.columns = list(column_names)
        imputer = GAINImputer()
        imputer.fit(X_missing, target_columns=list(column_names))
        #
        X_missing = pd.DataFrame(df.numpy())
        X_missing.columns = list(column_names)
        X_imputed, _ = imputer.transform(X_missing)
        #
        return X_imputed

    def impute(self, df, column_names=None, df_train=None):
        if EXCEPTION_RETRY:
            # sometimes there is a random dimension error with some tensors
            # possible caused by NaN's in the computation. Because the algorithm
            # is not deterministic a retry sometimes helps
            try:
                return self._run_impute(df, df_train=df_train, column_names=column_names)
            except Exception as e:
                print(f'*** Caught Exception {e}: retrying ....')
                return self._run_impute(df, df_train=df_train, column_names=column_names)
        else:
            return self._run_impute(df, column_names=column_names, df_train=df_train)

if __name__ == "__main__":
    Imputer().random_test()
