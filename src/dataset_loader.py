import sys
import os
import json
import wget
import numpy as np
import pandas as pd
from scipy.io import arff, loadmat
import ssl

from collections import namedtuple

from dataset_table import ALL_DATASETS

DatasetTuple = namedtuple("DatasetTuple", ["descr", "df"])

# DEFAULT_CACHE = "../DATASET_CACHE"
DEFAULT_CACHE = os.path.join("..", "DATASET_CACHE")

# creata a dummy certificate
ssl._create_default_https_context = ssl._create_unverified_context

class DatasetLoader:

    def __init__(self, cache=DEFAULT_CACHE, datasets=ALL_DATASETS):
        self.cache = cache
        if datasets is not None:
            self.dataset_table = dict()
            for dataset_json in datasets:
                self.dataset_table[dataset_json['name']] = dataset_json
        else:
            self.dataset_table = None

    def dataset_names(self, source=None):
        res = []
        for ds in self.dataset_table.values():
            if source is not None and ds["source"] != source:
                continue
            if "disabled" in ds and ds["disabled"]:
                continue
            res.append(ds["name"])
        return res

    def load_dataset(self, json_descr):
        name = json_descr["name"]
        dsd = get_dataset_descr(name)
        extension = json_descr["kind"]
        filename = os.path.join(self.cache,name+"."+extension)
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            if extension == "arff":
                csv_filename = os.path.join(self.cache,name+"."+"csv")
                if os.path.isfile(csv_filename) and os.access(csv_filename, os.R_OK):
                    return csv_filename
                else:
                    # remove .arff file and start verything new new
                    os.remove(filename)
            else:
                return filename
        if "url" in json_descr:
            url = json_descr["url"]
            # let the exception crash, this makes debugging easyer
            wget.download(url, filename)
            if extension == "arff":
                print(f'+ Loading arff file: {filename}')
                data = arff.loadarff(filename)
                df = pd.DataFrame(data[0])
                filename = os.path.join(self.cache,name+"."+"csv")
                #
                # INCOMPLETE, FIX this conversion duplicate code
                if True: # strip quotes
                    df.columns = [c.strip("'") for c in df.columns]
                for c in dsd.decode_columns():
                    df[c] = df[c].str.decode('utf-8') 
                if "numeric_columns" in json_descr:
                    # print(f'COLUMNS={df.columns}')
                    for c in json_descr["numeric_columns"]:
                        df[c] = pd.to_numeric(df[c]) 
                df.to_csv(filename, index=False)
            else:
                os.remove(filename)
                raise Exception("INCOMPLETE")
        elif "path" in json_descr and extension == "csv":
            path = json_descr["path"]
            src = os.path.join(self.cache,path)
            df = pd.read_csv(src, sep=',')
            # INCOMPLETE, FIX this conversion duplicate code
            if "numeric_columns" in json_descr:
                for c in json_descr["numeric_columns"]:
                    # print(f'Column: {c} = {df[c]}')
                    df[c] = pd.to_numeric(df[c]) 
            filename = os.path.join(self.cache,name+"."+"csv")
            df.to_csv(filename)
        elif "path" in json_descr and extension == "mat":
            path = json_descr["path"]
            src = os.path.join(self.cache,path)
            # print(f"* Loading .mat file, location={src}")
            data = loadmat(src)
            #
            df = pd.DataFrame(data['X'])
            df_y = pd.DataFrame(data['Y'])
            print(f'DF={df}')
            print(f'DF-Y={df_y}')

            raise Exception("XXXXXXXXXX")



    
            # print(f'{mat}')
            mat_items = {k:v for k, v in mat.items() if k[0] != '_'}
            # print(f'+ MAT items: {mat_items}')
            df = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat_items.items()})
            # print(f'+ COLUMNS = {df.columns}')
            # INCOMPLETE, FIX this conversion duplicate code
            if "numeric_columns" in json_descr:
                for c in json_descr["numeric_columns"]:
                    # print(f'Column: {c} = {df[c]}')
                    df[c] = pd.to_numeric(df[c]) 
            print(f'DF={df}')
            raise Exception("INCOMPLETE")
            filename = os.path.join(self.cache,name+"."+"csv")
            df.to_csv(filename)
        elif extension == "generated":
            df = self.generate_dataset(json_descr["numeric_columns"], json_descr["catagorical_columns"], json_descr["label_column"], json_descr["rows"])
            filename = os.path.join(self.cache,name+"."+"csv")
            df.to_csv(filename, index=False)
        else:
            raise Exception("INCOMPLETE")
        return filename

    def generate_dataset(self, nc, cc, label_c, rows):
        ddf = {}
        start = 1
        np.random.seed(1001)
        for c in nc:
            ddf[c] = range(start,start+rows)
            start *= 10
        i = 0
        for c in cc:
            ev = ENUMS[i]
            ddf[c] = pd.Series(np.random.choice(ENUMS[i], rows))
            i += 1
        # ddf[label_c] = [chr(x) for x in range(ord('A'), ord('A')+rows)]
        ddf[label_c] = range(1000,1000+rows)
        res = pd.DataFrame(ddf)
        return res

    def load_csv_dataframe(self, json_descr):
        f = self.load_dataset(json_descr)
        df = pd.read_csv(f)
        return df

    def exists_dataset(self, name):
        return (name in self.dataset_table.keys())

    def get_dataset_json(self, name):
        if name in self.dataset_table:
            return self.dataset_table[name]
        else:
            raise Exception(f'UNKOWN DATASET "{name}"')

ENUMS = [
    ["AA", "BB", "CC"],
    ["PP", "QQ", "RR", "SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ"]
]

#
#
#

class DatasetDescriptor:

    def __init__(self, json):
        self.json = json

    def kind(self):
        if "kind" in self.json:
            return self.json["kind"]
        else:
            raise Exeption("MISSING KIND")

    def label_column(self):
        if "label_column" in self.json:
            return self.json["label_column"]
        return self.json["text_columns"][0] # crash when no text_columns

    def label_column_index(self, df):
        print(df.columns)
        print(self.label_column())
        return df.columns.get_loc(self.label_column())

    def regression_column(self):
        return self.label_column()

    def text_columns(self):
        if "text_columns" in self.json:
            return self.json["text_columns"]
        else:
            return []

    def numeric_columns(self):
        if "numeric_columns" in self.json:
            return self.json["numeric_columns"]
        else:
            return []

    def is_numeric_column(self, column_name):
        return column_name in self.numeric_columns()

    def catagorical_columns(self):
        if "catagorical_columns" in self.json:
            return self.json["catagorical_columns"]
        else:
            if "categorical_columns" in self.json:
                # raise Exception(f'BAD tag \"categorical_columns\" in json: {self.json}')
                return self.json["categorical_columns"]
            return []

    def decode_columns(self):
        if "decode_columns" in self.json:
            return self.json["decode_columns"]
        else:
            return []

    def feature_columns(self):
        return self.numeric_columns() + self.catagorical_columns()

    def feature_indices(self, df):
        return self.numeric_indices(df) + self.catagorical_indices(df)

    def numeric_indices(self, df):
        numeric_columns = self.numeric_columns()
        numeric_indices= [(df.columns.get_loc(c) if not isinstance(c, int) else c) for c in numeric_columns]
        return numeric_indices

    def catagorical_indices(self, df):
        catagorical_columns = self.catagorical_columns()
        catagorical_indices= [(df.columns.get_loc(c) if not isinstance(c, int) else c) for c in catagorical_columns]
        return catagorical_indices

    def nominal_indices(self, df):
        nominal_columns = self.text_columns()
        nominal_indices= [(df.columns.get_loc(c) if not isinstance(c, int) else c) for c in nominal_columns]
        if len(nominal_indices) != 1:
            raise Exception(f'Expect exactly 1 text_column in "text_columns" field of dataset "{self.json}"')
        return nominal_indices

    def dropna(self):
        if "dropna" in self.json:
            return self.json["dropna"].lower() == "true"
        else:
            return False

    def contains_missing(self):
        if "contains_missing" in self.json:
            return self.json["contains_missing"].lower() == "true"
        else:
            return False


dsl = DatasetLoader()

def get_dataset_names(source=None):
    return dsl.dataset_names(source=source)

def get_dataset_descr(name):
    return DatasetDescriptor(dsl.get_dataset_json(name))

def exists_dataset(name):
    return dsl.exists_dataset(name)

def get_dataset(name, original=True, minmax=False, encode_catagorical=False):
    descr = DatasetDescriptor(dsl.get_dataset_json(name))
    #
    if descr.kind() == "special":
        df = None
        if name == "hospital1":
            df  = pd.read_excel("hospital1.xlsx")
            # dsd =  get_dataset_descr("hospital1")
        elif name == "RNA1":
            df  = pd.read_csv("RNA1.csv")
            # dsd =  get_dataset_descr("rna")
        else:
            raise Exception(f'UNKOWN SPECIAL DATASET "{name}"')
        return DatasetTuple(descr=descr, df=df)
    #
    df = dsl.load_csv_dataframe(descr.json)
    if descr.dropna():
        pd.DataFrame.dropna(df, axis=0, inplace=True)
    if original:
        return DatasetTuple(descr=descr, df=df)
    else:
        raise Exception("INCOMPLETE")
        return DatasetTuple(descr=descr, df=df)

def inspect_column(kind, df, col):
    print(f'  - Column: {col} (kind={kind}, dtype={df[col].dtypes})')
    print(f'    - # of uniq non NaN values: {df[col].nunique()}')
    if df[col].nunique() < 100:
        print(f'    - # uniq values: {df[col].unique()}')
    else:
        print(f'    - # uniq values: TOO_MUCH')
    try :
        print(f'    - # of NaN values: {np.isnan(df[col]).sum()}')
    except Exception:
        print(f'    - # of NaN values: FAIL')
    # print(f'    - # dropna: {df[col].dropna()}')


