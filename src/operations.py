import sys
import datetime

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from sklearn import preprocessing

from dataset_loader import get_dataset
from dataset_imputer import impute_df, new_impute_df

from impute_utils import produce_NA, produce_NA_2
import impute_utils


VERBOSE = False


class Timer:

    def start(self):
        self.dt_start = datetime.datetime.now()

    def stop(self):
        self.dt_delta = datetime.datetime.now() - self.dt_start
        self._last_seconds = self.dt_delta.total_seconds()

    def last_seconds(self):
        return self._last_seconds


TIMER = Timer()


def scale_dataset(df, dsd):
    scaled_df = df.copy(deep=True)
    text_columns = dsd.text_columns()
    le = preprocessing.LabelEncoder()


def print_ds(df, name=None, verbose=VERBOSE):
    np.set_printoptions(threshold=sys.maxsize)
    if name is not None:
        print(f'+ Dataset: {name}:')
    print(df)

def load_ds(name, verbose=VERBOSE):
    dst = get_dataset(name)
    if verbose:
        print_ds(dst.df, name=name)
    df = normalize_ds(dst.df, dst.descr.numeric_columns(), dst.descr.label_column(), name=name, verbose=verbose) 
    return df, dst.descr


def normalize_ds(df, num_columns, y_column, name=None, verbose=VERBOSE):
    le = preprocessing.LabelEncoder()
    if not is_numeric_dtype(df[y_column]):
        if True:
            # labelencoder already sorts the keys by default
            df[y_column] = le.fit_transform(df[y_column].values)
        else:
            mapping = {k: v for v, k in enumerate(sorted(df[y_column].unique()))}
            df[y_column] = df[y_column].map(mapping)
    if verbose:
        print_ds(df, name=name)
    return df


def generate_missing(src_df, missing, mecha, random_seed, verbose=VERBOSE):
    if missing < 0.0  or missing > 1.0:
        raise Exception("missing value should be between 0.0 and 1.0)")
    # missing_columns = src_df.iloc[:,dsd.numeric_indices(src_df)].values
    missing_columns = src_df.values
    if mecha == "MAR" or mecha == "MCAR" or mecha == "MNAR":
        dflt_mecha = mecha
        dflt_opt   = impute_utils.DFLT_OPT
        dflt_p_obs = impute_utils.DFLT_P_OBS
        dflt_q     = impute_utils.DFLT_Q
    elif mecha == "MCAR-ISIL":
        dflt_mecha = "MCAR"
        dflt_opt   = impute_utils.DFLT_OPT
        dflt_p_obs = impute_utils.DFLT_P_OBS
        dflt_q     = impute_utils.DFLT_Q
    elif mecha == "MCAR2":
        dflt_mecha = "MCAR"
        dflt_opt   = impute_utils.DFLT_OPT
        dflt_p_obs = impute_utils.DFLT_P_OBS
        dflt_q     = impute_utils.DFLT_Q
    elif mecha == "MNAR-QUANTILE":
        dflt_mecha = "MNAR"
        dflt_opt   = "quantile"
        dflt_p_obs = impute_utils.DFLT_P_OBS
        dflt_q     = impute_utils.DFLT_Q
    else:
        raise Exception(f'UNKNOWN mecha "{mecha}"')
    mccar = produce_NA(
                missing_columns,
                p_miss=missing,
                mecha=dflt_mecha,
                random_seed=random_seed,
                opt=dflt_opt,
                p_obs=dflt_p_obs,
                q=dflt_q)
    df = pd.DataFrame(mccar['X_incomp'].numpy())
    df.columns = src_df.columns
    return df

def generate_missing_2(src_df, missing, mecha, random_seed, verbose=VERBOSE):
    if missing < 0.0  or missing > 1.0:
        raise Exception("missing value should be between 0.0 and 1.0)")
    # missing_columns = src_df.iloc[:,dsd.numeric_indices(src_df)].values
    missing_columns = src_df.values
    mccar= produce_NA_2(missing_columns, p_miss=missing, mecha=mecha, random_seed=random_seed)
    df = pd.DataFrame(mccar['X_incomp'].numpy())
    df.columns = src_df.columns
    return df

def impute(src_df, impute, df_train=None, random_state=None, normalized=False, verbose=VERBOSE):
    if random_state is not None:
        np.random.seed(random_state)
    imputed_df = impute_df(src_df, impute, df_train=df_train, normalized=normalized)
    if True:
        return imputed_df

def new_impute(src_df, impute, dsd, X_org=None, normalized=False, verbose=VERBOSE):
    imputed_df = new_impute_df(src_df, impute, dsd, X_org=X_org, verbose=verbose)
    if True:
        return imputed_df

def create_result_df(json_res):
    d = ['ds_name']
    df = pd.DataFrame([], columns=d)
    for res_line in json_res:
        dict_row = dict()
        for f, v in res_line.items():
            dict_row[f] = v
        df_row = pd.DataFrame([dict_row])
        df = pd.concat([df, df_row], ignore_index=True)
    return df


if __name__ == "__main__":
    print("MAIN_OPERATIONS")
