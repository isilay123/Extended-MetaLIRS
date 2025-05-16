from .imputer_base import ImputerBase

import numpy as np
from sklearn.impute import SimpleImputer


IMPUTE_NAME = "MEAN"


#
# MEAN imputer, documentation can be found at
#


class Imputer(ImputerBase):

    def __init__(self):
        super().__init__(IMPUTE_NAME)

    def impute(self, df, column_names=None, df_train=None):
        mean=SimpleImputer(missing_values=np.nan, strategy='mean')
        if df_train is None:
            impNorm = mean.fit(df)
        else:
            impNorm = mean.fit(df_train)
        res = mean.transform(df)
        return res


if __name__ == "__main__":
    Imputer().random_test()
