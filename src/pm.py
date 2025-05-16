# first set tensorflow in eager mode otherwise VAE and GAIN4 do not work
# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
#
import os
import sys
import warnings
import traceback 
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from analyze_Xy import analyze_Xy_features

from regressors import get_regressor


import numpy as np
import pandas as pd

import operations as op
from dataset_loader import get_dataset

from impute_utils import RMSE, MAE, ACCURACY, generate_html

IGNORE_ERRORS = True

NORMALIZE_NUMERIC = True
NORMALIZE_CATAGORICAL = True

USE_HYPERPAR = False

IMPUTE_TEST_FRAME = True

TEST_SIZE = 0.25
RANDOM_SEED = 1001

CACHED_IMPUTE_ID = ""
CACHED_IMPUTE_DF = None
CACHED_IMPUTE_TEST_DF = None
CACHED_IMPUTE_RES = None



def run_single(ds_name, missing, mecha, imputer, new_imputer, regressor_name, analyze_features, random_seed=RANDOM_SEED, use_hyperpar=USE_HYPERPAR, round_catagorical=False, verbose=False, ignore_errors=IGNORE_ERRORS):
    res = {'ds_name': ds_name, 'missing':missing, 'mecha':mecha,
           'imputer': imputer, 'new_imputer': new_imputer,
           'regressor':regressor_name, 'random_seed':random_seed}

    dsd, df = get_dataset(ds_name, original=True)

    print(f'+++ RUNNING: {res}', flush=True)

    if verbose:
        print(f'\n+ ORIGINAL DATASET =\n{df}')

    X = df[dsd.feature_columns()]
    y = df[dsd.label_column()].astype(np.float64)

    if NORMALIZE_CATAGORICAL:
        label_encoder = LabelEncoder()
        X_new = X.copy()
        for ccol in dsd.catagorical_columns():
            X_new[ccol] = label_encoder.fit_transform(X[ccol])
        X = X_new

        if verbose:
            print('')
            op.print_ds(X, name="X(after label_encoder)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_seed)
    
    if True:
        X_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)
        y_train = y_train.astype(np.float64)
        y_test = y_test.astype(np.float64)

    if verbose:
        print("\n+ Train/Test before min-max normalization")
        op.print_ds(X_train, name="X_train")
        op.print_ds(y_train, name="y_train")
        op.print_ds(X_test, name="X_test")
        op.print_ds(y_test, name="y_test")

    if NORMALIZE_NUMERIC:
        scaler = StandardScaler()
        X_train[dsd.numeric_columns()] = scaler.fit_transform(X_train[dsd.numeric_columns()])
        X_test[dsd.numeric_columns()] = scaler.transform(X_test[dsd.numeric_columns()])
        if verbose:
            print("\n+ Train/Test AFTER min-max normalization")
            op.print_ds(X_train, name="X_train")
            op.print_ds(y_train, name="y_train")
            op.print_ds(X_test, name="X_test")
            op.print_ds(y_test, name="y_test")

    if False:
        X_train.to_csv(f'{os.path.join("REGRESSOR_TEST","X_train_original.csv")}', index=False)


    if missing is not None:
        op.TIMER.start()
        X_train_missing = op.generate_missing(X_train, missing, mecha, random_seed)
        op.TIMER.stop()
        res['t_missing'] = op.TIMER.last_seconds()
        if verbose:
            print('')
            op.print_ds(X_train_missing, name="X_train_missing")

        if IMPUTE_TEST_FRAME:
            X_test_missing = op.generate_missing(X_test, missing, mecha, random_seed)
            if verbose:
                print('')
                op.print_ds(X_test_missing, name="X_test_missing")

    else:
        X_train_missing = X_train

 

    if missing is not None and imputer is not None:
        if verbose:
            print(f'\n+ START IMPUTING: imputer="{imputer}", dataset="{ds_name}", missing=.{missing}')
        global CACHED_IMPUTE_ID
        global CACHED_IMPUTE_DF
        global CACHED_IMPUTE_TEST_DF
        global CACHED_IMPUTE_RES
        cache_id = f'{ds_name}{missing}{mecha}{imputer}{new_imputer}{random_seed}'
        if cache_id == CACHED_IMPUTE_ID:
            print(f'USING CACHED impute file: {cache_id}')
            X_train_imputed = CACHED_IMPUTE_DF
            if IMPUTE_TEST_FRAME:
                X_test_imputed = CACHED_IMPUTE_TEST_DF
            res['t_impute'] = CACHED_IMPUTE_RES['t_impute']
            if 't_missing' in res:
                res['t_missing'] = CACHED_IMPUTE_RES['t_missing']
        else:
            op.TIMER.start()
            if new_imputer:
                X_train_imputed = op.new_impute(X_train_missing, imputer, dsd, X_org=X_train, verbose=verbose)
                if IMPUTE_TEST_FRAME:
                    X_test_imputed = op.new_impute(X_test_missing, imputer, dsd, X_org=X_test, verbose=verbose)
            else:
                X_train_imputed = op.impute(X_train_missing, imputer)
                if X_train_imputed is None:
                    res['imputer_error'] = "true"
                    return res
                if IMPUTE_TEST_FRAME:
                    X_test_imputed = op.impute(X_test_missing, imputer)
                    if X_test_imputed is None:
                        res['imputer_error'] = "true"
                        return res
            op.TIMER.stop()
            res['t_impute'] = op.TIMER.last_seconds()
            #
            if True:
                CACHED_IMPUTE_ID = cache_id
            CACHED_IMPUTE_DF = X_train_imputed
            if IMPUTE_TEST_FRAME:
                CACHED_IMPUTE_TEST_DF = X_test_imputed
            CACHED_IMPUTE_RES = res

        if verbose:
            print('')
            op.print_ds(X_train_imputed, name="IMPUTED RESULT")
            if IMPUTE_TEST_FRAME:
                print('')
                op.print_ds(X_test_imputed, name="IMPUTED TEST RESULT")

        if X_train_imputed is None:
            res['timedout'] = "True"
            return res
        else:
            if verbose: 
                print('\n+ compute scores of result soo far')
            if True:
                if len(dsd.numeric_columns()) > 0:
                    X_train_num = X_train[dsd.numeric_columns()]
                    X_train_missing_num = X_train_missing[dsd.numeric_columns()]
                    X_train_imputed_num = X_train_imputed[dsd.numeric_columns()]
                    #
                    num_mask = np.isnan(X_train_missing_num.values)
                    res['mask_size_num'] = np.count_nonzero(num_mask)
                    res['impute_rmse_num'] = RMSE(X_train_imputed_num.values, X_train_num.values, num_mask)
                    res['impute_mae_num'] = MAE(X_train_imputed_num.values, X_train_num.values, num_mask) 
                if len(dsd.catagorical_columns()) > 0:
                    X_train_cat = X_train[dsd.catagorical_columns()]
                    X_train_missing_cat = X_train_missing[dsd.catagorical_columns()]
                    X_train_imputed_cat = X_train_imputed[dsd.catagorical_columns()]
                    #
                    cat_mask = np.isnan(X_train_missing_cat.values)
                    res['mask_size_cat'] = np.count_nonzero(cat_mask)
                    res['impute_rmse_cat'] = RMSE(X_train_imputed_cat.values, X_train_cat.values, cat_mask)
                    #
                    # print(cat_mask)
                    # print(X_train_imputed_cat)
                    old_option = pd.options.mode.copy_on_write
                    pd.options.mode.copy_on_write = True
                    X_train_imputed_cat.loc[cat_mask] = X_train_imputed_cat.loc[cat_mask] \
                              .apply(round, axis='columns') \
                              .values
                    pd.options.mode.copy_on_write = old_option
                    # print(X_train_imputed_cat)
                    res['impute_acc_cat'] = ACCURACY(X_train_imputed_cat.values, X_train_cat.values, cat_mask)
                    # sys.exit(0)
                #
                #
                #
            mask = np.isnan(X_train_missing.values)
            res['mask_size'] = np.count_nonzero(mask)
            res['miss_perc'] = (np.count_nonzero(mask)/mask.size) * 100
            res['impute_rmse'] = RMSE(X_train_imputed.values, X_train.values, mask)
            res['impute_mae'] = MAE(X_train_imputed.values, X_train.values, mask) 
            
            if verbose: 
                print(json.dumps(res, indent=4))
    else:
        if verbose:
            print('\n+ SKIPPED IMPUTING')
        X_train_imputed = X_train_missing
        if IMPUTE_TEST_FRAME:
            X_test_imputed = X_test

    if verbose:
        print(f'\n+ running regressor: {regressor_name}')
    try:
        regressor = get_regressor(regressor_name)
        
        op.TIMER.start()
        regressor.fit(X_train_imputed, y_train)
        op.TIMER.stop()
        res['t_fit'] = op.TIMER.last_seconds()

        if analyze_features is not None and len(analyze_features) > 0:
            # print(f'\n+ Analyzing features: {analyze_features}')
            analyze_Xy_features(X_train_missing, y_train, dsd, analyze_features, res, verbose=verbose)

        if verbose:
            print(f'\n+ regressor finished: {regressor_name}')
        y_predict = regressor.predict_score(X_test, y_test, res, pfx='y_test')
        y_predict = regressor.predict_score(X_train_imputed, y_train, res, pfx='y_train')
        if IMPUTE_TEST_FRAME:
            y_predict = regressor.predict_score(X_test_imputed, y_test, res, pfx='y_test_imp')
        if verbose:
            print(f'\n+ scores for {regressor_name}')
        if False:
            generate_html('/Users/flokstra/df.html', X_train, X_train_missing, X_train_imputed, y_predict=y_predict, y_original=y_train, label="X_train/y-predict")
        res['regressor_error'] = False
    except Exception as e:
        traceback.print_exception(*sys.exc_info())
        if not ignore_errors:
            sys.exit(-1)
        else:
            res['regressor_error'] = True
    print('', end='', flush=True)
    return res


def _distributed_file(distributed_tag,seed,ds_name):
    return Path('.').joinpath("DISTRIBUTED_RES").joinpath(distributed_tag).joinpath(str(seed)).joinpath(f'{ds_name}.xlsx')

def _distributed_dataframe(distributed_tag,seed,ds_name):
    file = _distributed_file(distributed_tag,seed,ds_name)
    return pd.read_excel(file)

def distributed_res_exists(distributed_tag,seed,ds_name):
    file = _distributed_file(distributed_tag,seed,ds_name)
    return file.is_file()

def distributed_res_save(distributed_tag,seed,ds_name,ds_res):
    file = _distributed_file(distributed_tag,seed,ds_name)
    if True:
        print(f'+ saving distributed res: {file}')
    dir = file.parent
    dir.mkdir(parents=True, exist_ok=True)
    df_res = op.create_result_df(ds_res)
    df_res.to_excel(file)

PLAN_VERBOSE = False

SMALL_PLAN = {
    "SEED": RANDOM_SEED,
    "REPEAT": 2,
    "DATASETS": ["bupa1"],
    "MISSING": [0.5],
    "MECHA": ["MCAR"],
    "IMPUTER": ["MLP2"],
    "NEW_IMPUTER": [],
    "ANALYZE_FEATURES": ["MF1", "MF2", "MF3", "MF4", "MF5", "MF6", "MF7", "MF8", "MF9", "MF10", "MF11", "MF12", "MF13", "MF14", "MF15", "MF16", "MF17", "MF18", "MF19", "MF20", "MF21", "MF22", "MF23", "MF24", "MF25", "MF26", "MF27", "MF28", "MF29", "MF30", "MF31", "MF32", "MF33"
],   
   "REGRESSOR": ["MLPregressor","svregression","xgboost","ridgeregression"],
    "MISSING_REGRESSOR": ["xgboost", "lightgbm"]
}











if __name__ == "__main__":
    PLAN = SMALL_PLAN
    
   
    #MODE = "PLAN"
    MODE = "SINGLE"
   
    #
    if MODE == "PLAN":
        USING_WINDOWS = hasattr(sys, 'getwindowsversion')
        if USING_WINDOWS:
            hostname = 'WINDOWS'
        else:
            hostname = os.uname()[1]
        print(f'Running pm.py on "{hostname}", PLAN = {json.dumps(PLAN, indent=4)}')
        print(f'NORMALIZE_NUMERIC={NORMALIZE_NUMERIC}')
        print(f'NORMALIZE_CATAGORICAL={NORMALIZE_CATAGORICAL}')
        print(f'USE_HYPERPAR={USE_HYPERPAR}')
        print(f'IMPUTE_TEST_FRAME={IMPUTE_TEST_FRAME}')
        print('\nREGRESSOR get_params():')
        if True:
            print('CHECKING DATASETS:')
            for ds_name in PLAN["DATASETS"]:
                print(f'+ {ds_name}', end='')
                try:
                    dsd, df = get_dataset(ds_name, original=True)
                except Exception as e:
                    print(f'!!!!!! DATASET ERROR !!!!!!!')
                    print(f'DATASET_NAME: "{ds_name}"')
                    print(f'EXCEPTION: {type(e)}: {e}')
                    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    traceback.print_exception(*sys.exc_info())
                    sys.exit(-1)
                print(f'[{df.values.shape}]: ', end='')
                print('OK')
        for regressor_name in PLAN["REGRESSOR"]:
            regressor = get_regressor(regressor_name)
            print(f'\t{regressor_name}:\t{regressor.get_params()}')
        if "DISTRIBUTED_TAG" in PLAN:
            distributed_tag = PLAN["DISTRIBUTED_TAG"]
            print(f'DISTRIBUTED_TAG={distributed_tag}')
        else:
            distributed_tag = None
        print('')
        analyze_features = PLAN["ANALYZE_FEATURES"]
        if "REPEAT" in PLAN:
            repeat = PLAN["REPEAT"]
        else:
            repeat = 1
        res_all = []
        for seed in range(PLAN["SEED"], PLAN["SEED"]+repeat):
            for ds_name in PLAN["DATASETS"]:
                if not distributed_tag or (distributed_tag is not None and not distributed_res_exists(distributed_tag,seed,ds_name)):
                    ds_res = [] 
                    for missing in PLAN["MISSING"]:
                        if missing is None:
                            mecha = None
                            imputer = None
                            for regressor in PLAN["REGRESSOR"]:
                                res = run_single(ds_name, missing, mecha, imputer, None, regressor, analyze_features, random_seed=seed, verbose=PLAN_VERBOSE)
                                ds_res.append(res)
                        else:
                            for mecha in PLAN["MECHA"]:
                                for regressor in PLAN["MISSING_REGRESSOR"]:
                                    imputer = None
                                    res = run_single(ds_name, missing, mecha, imputer, None, regressor, analyze_features, random_seed=seed, verbose=PLAN_VERBOSE)
                                    ds_res.append(res)
                                for imputer in PLAN["IMPUTER"]:
                                    for regressor in PLAN["REGRESSOR"]:
                                        res = run_single(ds_name, missing, mecha, imputer, False, regressor, analyze_features, random_seed=seed, verbose=PLAN_VERBOSE)
                                        ds_res.append(res)
                                for imputer in PLAN["NEW_IMPUTER"]:
                                    for regressor in PLAN["REGRESSOR"]:
                                        res = run_single(ds_name, missing, mecha, imputer, True, regressor, analyze_features, random_seed=seed, verbose=PLAN_VERBOSE)
                                        ds_res.append(res)
                    if distributed_tag is not None:
                        distributed_res_save(distributed_tag,seed,ds_name,ds_res)
                    for res in ds_res:
                        res_all.append(res)
                else:
                    print(f'* skipping precomputed {distributed_tag}/{seed}/{ds_name}')
            df_all = op.create_result_df(res_all)
            df_all.to_excel('./RESULTS/last_pm_result.xlsx')
            # print(f'{df_all}')
            print(f'{res_all}')
            # print(f'{json.dumps(df_all,indent=4)}')
            print(f'*** Finished according to plan')
    elif MODE == "SINGLE":
        VERBOSE = True
        PLAN = {
          
            "DATASET":  "bupa1",
            "MISSING": 0.05,
            "MECHA": "MAR",
         
            "IMPUTER": "MEAN",
          
            "ROUND_CATAGORICAL" : False,
          
            "USE_NEW_IMPUTER": False,
            
            "ANALYZE_FEATURES" : ["MF2"],       
            "REGRESSOR": "lightgbm",
          
            "SEED": RANDOM_SEED
        }
        new_imputer = "USE_NEW_IMPUTER" in PLAN and PLAN["USE_NEW_IMPUTER"] == True
        res = run_single(PLAN["DATASET"], PLAN["MISSING"], PLAN["MECHA"], PLAN["IMPUTER"], new_imputer, PLAN["REGRESSOR"], PLAN["ANALYZE_FEATURES"], random_seed=PLAN["SEED"], round_catagorical=PLAN["ROUND_CATAGORICAL"], use_hyperpar=USE_HYPERPAR, verbose=VERBOSE)
        #
        try:
            print(json.dumps(res, indent=4))
        except Exception as e:
            print(f'JSON*NOT*SERIALIZABLE, RAW_DATA={res}')
    elif MODE == "COLLECT": # collect dataframes of distributed plan
        if "DISTRIBUTED_TAG" in PLAN:
            distributed_tag = PLAN["DISTRIBUTED_TAG"]
        else:
            raise Exception("Trying to collect non-distributed plan")
        print(f'* Collecting distributed PLAN: "{distributed_tag}"')
        if "REPEAT" in PLAN:
            repeat = PLAN["REPEAT"]
        else:
            repeat = 1
        res_df = None
        missing = 0
        for seed in range(PLAN["SEED"], PLAN["SEED"]+repeat):
            for ds_name in PLAN["DATASETS"]:
                if distributed_res_exists(distributed_tag,seed,ds_name):
                    print(f'+ collecting {distributed_tag}/{seed}/{ds_name}')
                    df = _distributed_dataframe(distributed_tag,seed,ds_name)
                    if res_df is None:
                        res_df = df
                    else:
                        res_df = pd.concat([res_df, df], ignore_index=True)
                else:
                    missing += 1
                    print(f'+ **MISSING* {distributed_tag}/{seed}/{ds_name}')
        if res_df is not None:
            if missing != 0:
                print(f'! WARNING: {missing} frames were missing')
            save_file = './RESULTS/last_collect_result.xlsx'
            print(f'* Saving file to: {save_file}')
            res_df.to_excel(save_file)
        else:
            print(f'! ERROR: nothing to collect')
        sys.exit(0)
    #
    else:
        raise Exception(f'pm: unknown mode \"{MODE}\"')
