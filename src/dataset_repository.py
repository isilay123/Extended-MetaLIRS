import glob
from pathlib import Path
import shutil
import json

import pandas as pd

from dataset_loader import get_dataset_names, get_dataset, get_dataset_descr
from dataset_transformations import scale_dataset, inverse_scale_dataset, generate_missing_values, impute_dataset, TIMER


DEFAULT_ROOT = Path("../DATASET_REPOSITORY/root")

VERBOSE = True

IMPUTE_TIMING_TAG = "impute_time"
MISSING_TIMING_TAG = "missing_time"

class Timeout(Exception):
    "timeout exception for imputers"
    pass


class DatasetRepository:

    def __init__(self, root=DEFAULT_ROOT):
        self.root = root
        self.build_verbose = False

    def clear_all(self):
        if self.root.is_dir():
            shutil.rmtree(self.root)

    def clear(self, datasets=None, mechas=None, imputers=None):
        dirs_to_rm = []
        if datasets is not None:
            for dataset in datasets:
                dirs_to_rm.append(self.d_src(dataset))
        elif mechas is not None:
            for mecha in mechas:
                dirs = glob.glob(str(DEFAULT_ROOT) + f'/*/*/{mecha}')
                dirs_to_rm.extend(dirs)
        elif imputers is not None:
            for imputer in imputers:
                dirs = glob.glob(str(DEFAULT_ROOT) + f'/*/*/*/*/{imputer}')
                dirs_to_rm.extend(dirs)
        if True:
            print('+ Clearing directories:')
            for d in dirs_to_rm:
                print(f'+ CLEAR: {d}')
                shutil.rmtree(d)

    #
    #
    #

    def d_src(self, name):
        return self.root.joinpath(name)

    def f_src(self, name, scaled):
        if scaled:
            return self.d_src(name).joinpath(f'{name}_scaled.csv')
        else:
            return self.d_src(name).joinpath(f'{name}.csv')

    def d_missing(self, name, missing, mecha, seed):
        return self.d_src(name).joinpath(f'missing-{missing}').joinpath(f'{mecha}').joinpath(f'seed-{seed}')

    def f_missing(self, name, missing, mecha, seed, scaled):
        if scaled:
            return self.d_missing(name, missing, mecha, seed).joinpath(f'{name}_scaled.csv')
        else:
            return self.d_missing(name, missing, mecha, seed).joinpath(f'{name}.csv')

    def d_imputed(self, name, missing, mecha, seed, impute):
        return self.d_missing(name, missing, mecha, seed).joinpath(f'{impute}')

    def f_imputed(self, name, missing, mecha, seed, impute, scaled):
        if scaled:
            return self.d_imputed(name, missing, mecha, seed, impute).joinpath(f'{name}_scaled.csv')
        else:
            return self.d_imputed(name, missing, mecha, seed, impute).joinpath(f'{name}.csv')

    #
    #
    #

    def read_csv_as_df(self, f_path):
        res = pd.read_csv(f_path)
        return res

    def write_df_as_csv(self, df, f_path, log):
        d_path = f_path.parent
        d_path.mkdir(parents=True, exist_ok=True)
        if df is not None:
            if self.build_verbose:
                print(f'-> {f_path}')
                # print(df)
            df.to_csv(f_path, index=False)
        #
        if log is not None:
            f_log_path = str(f_path).replace('.csv', '.log')
            with open(f_log_path, "w") as logfile:
                logfile.write(json.dumps(log, indent=4))

    def get_src(self, name, scaled):
        f = self.f_src(name, scaled)
        if f.is_file():
            return self.read_csv_as_df(f)
        else:
            return self.gen_src(name, scaled)

    def get_missing(self, name, missing, mecha, seed, scaled):
        f = self.f_missing(name, missing, mecha, seed, scaled)
        if f.is_file():
            return self.read_csv_as_df(f)
        else:
            return self.gen_missing(name, missing, mecha, seed, scaled)

    def get_imputed(self, name, missing, mecha, seed, impute, scaled):
        f = self.f_imputed(name, missing, mecha, seed, impute, scaled)
        if f.is_file():
            return self.read_csv_as_df(f)
        else:
            f_log = f.parent.joinpath(f'{name}.log')
            if f_log.is_file():
                # must be a timed_out op
                raise Timeout()
            return self.gen_imputed(name, missing, mecha, seed, impute, scaled)

    def get_timings(self, name, missing, mecha, seed, impute):
        d_impute = self.d_imputed(name, missing, mecha, seed, impute)
        d_missing = d_impute.parent
        f_impute = d_impute.joinpath(f'{name}.log')
        f_missing = d_missing.joinpath(f'{name}.log')
        #
        with open(f_impute, 'r') as myfile:
            impute_data = json.loads(myfile.read())
        with open(f_missing, 'r') as myfile:
            missing_data = json.loads(myfile.read())
        return {IMPUTE_TIMING_TAG : impute_data[IMPUTE_TIMING_TAG], MISSING_TIMING_TAG : missing_data[MISSING_TIMING_TAG]}

    def gen_src(self, name, scaled):
        log = dict()
        log["status"] = "OK"
        dataset_tuple = get_dataset(name)
        cache_df = dataset_tuple.df
        dsd = get_dataset_descr(name)
        df = pd.DataFrame()
        for numeric_col in dsd.numeric_columns():
            df = pd.concat((pd.DataFrame(df), cache_df[numeric_col]), axis=1)
        for text_col in dsd.text_columns():
            df = pd.concat((pd.DataFrame(df), cache_df[text_col]), axis=1)
        self.write_df_as_csv(df, self.f_src(name, False), log)
        #
        scaled_df = scale_dataset(df, dataset_tuple.descr)
        self.write_df_as_csv(scaled_df, self.f_src(name, True), None)
        #
        return scaled_df if scaled else df

    def gen_missing(self, name, missing, mecha, seed, scaled):
        log = dict()
        log["status"] = "OK"
        src_df = self.get_src(name, True)  # get the scaled version of the dataset
        dsd = get_dataset_descr(name)
        df = generate_missing_values(src_df, missing, mecha, seed, dsd)
        log[MISSING_TIMING_TAG] = TIMER.last_seconds()
        self.write_df_as_csv(df, self.f_missing(name, missing, mecha, seed, True), None)
        #
        unscaled_src = self.get_src(name, False)
        inverse_scaled_df = inverse_scale_dataset(df, unscaled_src, dsd)
        self.write_df_as_csv(inverse_scaled_df, self.f_missing(name, missing, mecha, seed, False), log)
        return df if scaled else inverse_scaled_df

    def gen_imputed(self, name, missing, mecha, seed, impute, scaled):
        log = dict()
        log["status"] = "OK"
        src_df = self.get_missing(name, missing, mecha, seed, True)
        dsd = get_dataset_descr(name)
        df = impute_dataset(src_df, impute, get_dataset_descr(name))
        if df is None:
            log["status"] = "TIMED_OUT"
            self.write_df_as_csv(None, self.f_imputed(name, missing, mecha, seed, impute, False), log)
            raise Timeout()
        else:
            log[IMPUTE_TIMING_TAG] = TIMER.last_seconds()
            self.write_df_as_csv(df, self.f_imputed(name, missing, mecha, seed, impute, True), None)
            #
            unscaled_src = self.get_src(name, False)
            inverse_scaled_df = inverse_scale_dataset(df, unscaled_src, dsd)
            self.write_df_as_csv(inverse_scaled_df, self.f_imputed(name, missing, mecha, seed, impute, False), log)
            return df if scaled else inverse_scaled_df

    def build_tree(self, datasets, percentages, mechas, base_seed, repeat, imputers, verbose=False):
        self.build_verbose = verbose
        for dataset in datasets:
            for percentage in percentages:
                for mecha in mechas:
                    for seed in range(base_seed, base_seed+repeat, 1):
                        try:
                            for imputer in imputers:
                                self.get_imputed(dataset, percentage, mecha, seed, imputer, False)
                        except Timeout:
                            if self.build_verbose:
                                print(f'-> Timeout on: {dataset}, {percentage}, {mecha}, {seed}, {imputer}')



