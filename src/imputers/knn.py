from .imputer_base import ImputerBase

from sklearn.impute import KNNImputer


IMPUTE_NAME = "KNN"

#
# KNN imputer, documentation can be found at
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
#


class Imputer(ImputerBase):

    def __init__(self):
        super().__init__(IMPUTE_NAME)

    def impute(self, df, column_names=None, df_train=None):
        #myKNN = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
        myKNN = KNNImputer()
        # impNorm = myKNN.fit(df)
        if df_train is None:
            myKNN.fit(df)
        else:
            myKNN.fit(df_train)
        res = myKNN.transform(df)
        #
        return res


if __name__ == "__main__":
    Imputer().random_test()
