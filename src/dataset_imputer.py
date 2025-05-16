import os
import sys
import importlib
import signal
import traceback
from contextlib import contextmanager
import torch

import numpy as np
import pandas as pd

from dataset_loader import get_dataset
from impute_utils import produce_NA
from impute_utils import RMSE, MAE

IMPUTE_TIME_OUT = 3600 # one hour timeout
USING_WINDOWS = hasattr(sys, 'getwindowsversion')
# USE_TIMEOUT = not USING_WINDOWS
USE_TIMEOUT = not USING_WINDOWS


loaded_imputers = dict()

new_loaded_imputers = dict()


def _get_imputer(name):
    name = name.lower()
    if name not in loaded_imputers:
        try:
            imputer_module = importlib.import_module(f'imputers.{name}')
        except Exception as e:
            print(f'!Fail to load imputer module {name}: {e}')
            print(f'!Make sure python file {os.path.join("imputers",name+".py")} exists and is correct.')
            print(f'!TIP: run "python -m imputers.{name}" from top level python directory to check')
            sys.exit(-1)
        try:
            imputer_class = getattr(imputer_module, "Imputer")
        except Exception as e:
            print(f'!Imputer exception: {e}')
            print('!Make sure there is an "Impute" class in the module')
            sys.exit(-1)
        try:
            loaded_imputers[name] = imputer_class()
        except Exception as e:
            print(f'!Imputer Class creator exception: {e}')
            print(f'!Class Imputer in {name}.py is not a proper Imputer class')
            sys.exit(-1)
    return loaded_imputers[name]

def _new_get_imputer(name):
    name = name.lower()
    if name not in new_loaded_imputers:
        try:
            imputer_module = importlib.import_module(f'new_imputers.{name}')
        except Exception as e:
            print(f'!Fail to load imputer module {name}: {e}')
            print(f'!Make sure python file {os.path.join("imputers",name+".py")} exists and is correct.')
            print(f'!TIP: run "python -m imputers.{name}" from top level python directory to check')
            sys.exit(-1)
        try:
            imputer_class = getattr(imputer_module, "Imputer")
        except Exception as e:
            print(f'!Imputer exception: {e}')
            print('!Make sure there is an "Impute" class in the module')
            sys.exit(-1)
        try:
            new_loaded_imputers[name] = imputer_class()
        except Exception as e:
            print(f'!Imputer Class creator exception: {e}')
            print(f'!Class Imputer in {name}.py is not a proper Imputer class')
            sys.exit(-1)
    return new_loaded_imputers[name]


class ImputeTimeout(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise ImputeTimeout("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def impute_df(df, imputer_name, df_train=None, normalized=False):
    src_tensor = torch.tensor(df.values)
    if df_train is not None:
        df_train = torch.tensor(df_train.values)
    interrupted = False
    imputer = _get_imputer(imputer_name)
    if normalized:
        imputer.skip_normalization()
    try:
        if USE_TIMEOUT:
            try:
                with time_limit(IMPUTE_TIME_OUT):
                    if df_train is not None:
                        try:
                            imputed_tensor = imputer.impute(src_tensor, column_names=df.columns, df_train=df_train)
                        except:
                            traceback.print_exc()
                            print(f'# IMPUTE FAILED: {imputer_name}')
                            print(f'# Probably caused by implementation without df_train implementation')
                            sys.exit(-1)
                    else:
                        imputed_tensor = imputer.impute(src_tensor, column_names=df.columns)
            except ImputeTimeout:
                interrupted = True
        else:
            if df_train is not None:
                try:
                    imputed_tensor = imputer.impute(src_tensor, column_names=df.columns, df_train=df_train)
                except:
                    traceback.print_exc()
                    print(f'# IMPUTE FAILED: {imputer_name}')
                    print(f'# Probably caused by implementation without df_train implementation')
                    sys.exit(-1)
            else:
                imputed_tensor = imputer.impute(src_tensor, column_names=df.columns)
    except:
        # a general exception during impute
        traceback.print_exc()
        interrupted = True
    if interrupted:
        return None
    else:
        imputed_df = pd.DataFrame(imputed_tensor)
        imputed_df.columns = df.columns
        #
        return imputed_df

def new_impute_df(src_df, imputer_name, dsd, X_org=None, verbose=False):
    interrupted = False
    imputer = _new_get_imputer(imputer_name)
    try:
        if USE_TIMEOUT:
            try:
                with time_limit(IMPUTE_TIME_OUT):
                    imputed_df = imputer.impute(src_df, dsd, X_org=X_org, verbose=verbose)
            except ImputeTimeout:
                interrupted = True
        else:
            imputed_df = imputer.impute(src_df, dsd, X_org=X_org, verbose=verbose)
    except:
        # a general exception during impute
        traceback.print_exc()
        interrupted = True
    if interrupted:
        return None
    else:
        return imputed_df

