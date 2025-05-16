import datetime

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from dataset_imputer import impute_df

from impute_utils import produce_NA, print_df

VERBOSE = True


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
    for text_col in text_columns:
        scaled_df[text_col] = le.fit_transform(df[text_col].values)
    minmax_scaler = MinMaxScaler()
    num_df = scaled_df.iloc[:,dsd.numeric_indices(df)]
    num_df_scaled = minmax_scaler.fit_transform(num_df)
    nrmData = pd.DataFrame(num_df_scaled)
    for text_col in text_columns:
        nrmData = pd.concat((pd.DataFrame(nrmData),scaled_df[text_col]), axis=1)
    scaled_df = pd.DataFrame(nrmData)
    scaled_df.columns = df.columns
    #
    return scaled_df

def inverse_scale_dataset(df_src, df_org, dsd):
    minmax_scaler = MinMaxScaler()
    org_num_df = df_org.iloc[:,dsd.numeric_indices(df_org)]
    minmax_scaler.fit_transform(org_num_df)
    src_num_df = df_src.iloc[:,dsd.numeric_indices(df_src)]
    inverse_src_num_df = minmax_scaler.inverse_transform(src_num_df)
    #
    res_df = pd.DataFrame(inverse_src_num_df)
    for text_col in dsd.text_columns():
        res_df = pd.concat((pd.DataFrame(res_df),df_org[text_col]), axis=1)
    res_df.columns = df_org.columns
    #
    return res_df

def generate_missing_values(src_df, missing, mecha, seed, dsd):
    if missing < 0  or missing > 100:
        raise Exception("missing value should be between 0 and 100)")
    missing_columns = src_df.iloc[:,dsd.numeric_indices(src_df)].values
    TIMER.start()
    mccar= produce_NA(missing_columns, p_miss=missing/100, mecha=mecha, random_seed=seed)
    TIMER.stop()
    df = pd.DataFrame(mccar['X_incomp'].numpy())
    #
    for text_col in dsd.text_columns():
        df = pd.concat((pd.DataFrame(df),src_df[text_col]), axis=1)
    #
    df.columns = src_df.columns
    return df

def impute_dataset(src_df, impute, dsd):
    columns_to_impute = src_df.iloc[:,dsd.numeric_indices(src_df)]
    TIMER.start()
    imputed_columns = impute_df(columns_to_impute, impute)
    TIMER.stop()
    if imputed_columns is None:
        return None
    else:
        df = pd.DataFrame(imputed_columns)
        for text_col in dsd.text_columns():
            df = pd.concat((pd.DataFrame(df),src_df[text_col]), axis=1)
        df.columns = src_df.columns
        #
        return df
