from .imputer_base import ImputerBase

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


IMPUTE_NAME = "MISSFOREST"


#
# MISSFOREST imputer, documentation can be found at
#


class Imputer(ImputerBase):

    def __init__(self):
        super().__init__(IMPUTE_NAME)

    def impute(self, df, column_names=None, df_train=None):
        #
        if df_train is None:
            df_train = df
        #
        rfr = RandomForestRegressor(
            # We tuned the hyperparameters of the RandomForestRegressor to get a good
            # enough predictive performance for a restricted execution time.
            #n_estimators=200, previous until 29.01.2025
            n_estimators=50,
            max_depth=10,
            #max_depth=10,
            #bootstrap=True,
            #max_samples=0.5,
            #n_jobs=-1,
            random_state=0,
        )
        # tolerances = (1e-3, 1e-1, 1e-1, 1e-2)
        # iter_imp = IterativeImputer(random_state=0, estimator=rfr, max_iter=50)
        iter_imp = IterativeImputer(random_state=0, estimator=rfr, max_iter=50)
        iter_imp.fit(df_train)
        res = iter_imp.transform(df)
        return res


if __name__ == "__main__":
    Imputer().random_test()
