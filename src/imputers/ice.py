from .imputer_base import ImputerBase

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


IMPUTE_NAME = "ICE"


#
# ICE imputer, documentation can be found at
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
#


class Imputer(ImputerBase):

    def __init__(self):
        super().__init__(IMPUTE_NAME)

    def impute(self, df, column_names=None, df_train=None):
        ice_mean = IterativeImputer(random_state=0, max_iter=50)
        #
        if df_train is None:
            df_train = df
        #
        ice_mean.fit(df_train)
        res = ice_mean.transform(df)
        #
        return res


if __name__ == "__main__":
    Imputer().random_test()
