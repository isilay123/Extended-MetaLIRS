import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor


MIN_TRAIN = 2
MAX_TRAIN = 250
NNAN = 'n_nan'


def simple_impute(df, global_mean):
    """ simple imputer to repair training dataset
    """
    mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    for col in df.columns:
        if df[col].isnull().all():
            # print(f'{col} : contains only NaN, dflt={global_mean[col]}')
            df[col].fillna(value=global_mean[col], inplace=True)
    mean.fit(df)
    res = mean.transform(df)
    res_df = pd.DataFrame(res)
    res_df.columns = df.columns
    return res_df


class MLPImputer:

    def __init__(self, train_all=False, verbose=False):
        self.train_all = train_all
        self.verbose = verbose
        self._fitted = False
        if verbose:
            stat_log = []

    def column_classifier(self, src_df, target_c, feature_c):
        """ function computes an MLP classifier on target_c columns by the
            feature columns. It dos so by selecting the 'best' candidate
            rows with the least number of NaN's. The remaining NaN's in
            the traning set a guessed by a simple imputer.
        """
        # make a real copy of the input Pandas DataFrame and add a columns
        # containing the number of NaN in every feature row
        raw_df = src_df.copy()
        raw_df[NNAN] = src_df[feature_c].isna().sum(1)
        if self.verbose:
            print(f'RAW_DF={raw_df}')
        df = raw_df
        #
        # filter out all target values which are not a NAN
        global_mean = df.mean(axis=0).to_dict()
        df = df[df[target_c].notnull()]
        # sort it by the least number of NaN per row
        df = df.sort_values(by=[NNAN], ascending=True)
        # now slice it to create the training set. A little bit more
        # intelligence could be used here.
        if not self.train_all:
            df = df[:MAX_TRAIN]
        if self.verbose:
            print(df)
        if df.shape[0] < MIN_TRAIN:
            raise Exception("INCOMPLETE: training set too small")
        # now impute the training set with a simple imputer to make sure
        # there are no NaN's anymore in the training data
        imp_df = simple_impute(df, global_mean)
        if self.verbose:
            print(imp_df)
        # create the final X, y used the create the classifier. The y is
        # a bit weird but this is how it should be done. Other solutions
        # caused an "unknown label" error.
        X_train = imp_df[list(feature_c)]
        # y_train = imp_df[c].values.flatten()
        y_train = np.asarray(imp_df[target_c], dtype=np.float64)
        # y_train = np.asarray(imp_df[target_c], dtype="|S6")
        #
        if self.verbose:
            print(f'X_train={X_train}')
            print(f'y_train={y_train}')
        # return the target column classifier
        return MLPRegressor(random_state=1, max_iter=2000).fit(X_train, y_train)
#max_iter=300
    def fit(self, df):
        """ the classifier fit() function which computes an MLP classifier for
            every column and stores it in the column classifiers.
        """
        # store the column name of the entire frame
        self.columns = columns = list(df.columns)
        # create the column classifier table
        column_classifiers = {}
        # compute the mean value of every column for later use
        self.column_mean = df.mean(axis=0).to_dict()
        # now create the classifier for every column, the reaming columns are
        # the feature columns
        for target_c in columns:
            feature_c = columns.copy()
            feature_c.remove(target_c)
            column_classifiers[target_c] = self.column_classifier(df, target_c, feature_c)
        self.column_classifiers = column_classifiers
        self._fitted = True

    def nan2colmean(self, df):
        """ this function has as parameter a frame with feature columns during transform().
            It check is there are NaN values in the feature row and replaces them by the
            column 'mean' computed during fit()
        """
        missing_mask = df.isna()
        if missing_mask.sum().sum() > 0:
            res_df = df.fillna(self.column_mean)
            return res_df
        else:
            return df

    def transform(self, df):
        """ this function imputes column by column the dataset using the column classifiers
            computed with the column_classifier() method. Feature values which are NaN are
            replace by the mean value of the column
        """
        if not self._fitted:
            raise Exception("NOT FITTED")
        columns = self.columns
        cclf = self.column_classifiers
        for target_c in columns:
            clf = cclf[target_c]
            other_c = columns.copy()
            other_c.remove(target_c)
            missing_mask = df[target_c].isna()
            n_missing = missing_mask.sum()
            if n_missing > 0:
                # the cast as float64 is because the dtype="|S6" for the y_train
                df.loc[missing_mask, target_c] = clf.predict(self.nan2colmean(df.loc[missing_mask, other_c])).astype(np.float64)
        return df


def load_iris_with_nan(perc, random_seed=1001):
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    np.random.seed(random_seed)
    for col in df.columns:
        df.loc[df.sample(frac=perc).index, col] = np.nan
    return df


if __name__ == "__main__":
    df = load_iris_with_nan(0.2)

    print(f'INPUT DataFrame:\n{df}')
    mlpi = MLPImputer(verbose=False)
    mlpi.fit(df)
    imputed_df = mlpi.transform(df)
    print(f'IMPUTED DataFrame:\n{imputed_df}')
