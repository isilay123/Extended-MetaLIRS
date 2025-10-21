import sys
import json
import datetime
import pandas as pd
import numpy as np
import statistics
from dataset_loader import DatasetLoader
from impute_utils import print_df
import analyze
from collections import namedtuple
from impute_utils import RMSE, MAE
from dataset_repository import DatasetRepository, Timeout
from dataset_loader import get_dataset_names, get_dataset_descr
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import nan_euclidean_distances
from scipy.stats import spearmanr
import operations as op
from scipy.stats import zscore
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from sklearn.model_selection import LeaveOneOut







Xy_tuple = namedtuple("Xy_tupe", ["X", "y"])





def percentile25(data):
  
    percentile_25 = np.percentile(data, 25)
    return percentile_25

def percentile75(data):
  
    percentile_75 = np.percentile(data, 75)
    return percentile_75

#For data complexity measures, see https://github.com/aclorena/ComplexityRegression/blob/master/measures.r 
#MF1 corresponds to the percentage of missing data; therefore, we did not include it in the code.
#We will add MF20-MF25 meta features (data complexity-based meta-features). If you want to apply them in the meantime, please check out these sources. Please see this sources:https://github.com/aclorena/ComplexityRegression/blob/master/measures.r
###############################################################################################


def pr(df):
    print(f'PR({type(df)})={df}')
    return df

def nan2zero(ds):
    return np.nan_to_num(ds, copy=True, nan=0.0)

  
FEATURE_TABLE = {
    "MF2":{
        "analyze":  lambda dt: len(np.unique(dt.y))
    },
    
    "MF3":{ 
        "analyze":  lambda dt: len(dt.X)
    },
    "MF4": {
        "use_numpy": True,
        "analyze":   lambda dt: dt.X.shape[1]/len(dt.X)
    },
   
    "MF5": {
        "analyze":   lambda dt:np.nanstd(dt.X).mean()
    },
    "MF6": {
        "analyze":   lambda dt: dt.y.std()
    },
    "MF7":{ 
        "analyze":  lambda dt: dt.X.shape[1] 
    },
    "MF8": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: dt.X.mean()
    },
    "MF9": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: dt.y.mean()
    },
    "MF10":{
        "use_numpy":True,
        "analyze":  lambda dt: np.nanmin(dt.X)
    },
    
    "MF11":{
        "use_numpy":True,
        "analyze":  lambda dt: np.nanmax(dt.X)
    },
    "MF12": {
        "analyze":   lambda dt: np.nanvar(dt.X).mean()
    },
    "MF13": {
        "analyze":   lambda dt: np.nanvar(dt.y) 
      
  },
    "MF14": {
        "analyze":   lambda dt: dt.X.corr().abs().mean().mean()
    },
    "MF15": {
        "analyze":   lambda dt: dt.X.corr().abs().std().mean()
    },
 
    "MF16": {
     
        "analyze":   lambda dt: dt.X.corrwith(dt.y).abs().mean().mean()
    },
    "MF17": {
      
        "analyze":   lambda dt: dt.X.corrwith(dt.y).max()
    },
    "MF18": {
        
        "analyze":   lambda dt: dt.X.corrwith(dt.y).min()
    },
    "MF19": {
        "analyze":   lambda dt: dt.X.cov().abs().mean().mean()
    },
    "MF20": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: l2(dt.X, dt.y, normalize=True)
    },
    "MF21": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: l3(dt.X, dt.y, normalize=False)
    },
    "MF22": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: s2(dt.X, dt.y, normalize=False)
    },
    "MF23": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: s3(dt.X, dt.y, normalize=False)
    },
    "MF24": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: s4(dt.X, dt.y, normalize=False)
    },
    "MF25": {
        "use_numpy": True,
        "nan2zero":  True,
        "analyze":   lambda dt: t2(dt.X, dt.y)
    },  
    
    "MF26": {
        "analyze":   lambda dt: dt.X.kurtosis().mean()
    },
    "MF27": {
        "analyze":   lambda dt: dt.X.skew().mean()
    },
    "MF28": {
        "analyze":   lambda dt: dt.y.skew()
    },
    "MF29": {
        "analyze":   lambda dt: dt.y.kurtosis()
    },
   "MF30": {
       "nan2zero":  True,
       "analyze":   lambda dt : np.percentile(dt.X,75)
   },
   "MF31": {
       "nan2zero":  True,
       "analyze":   lambda dt : np.percentile(dt.y,75)
   },
   "MF32": {
       "nan2zero":  True,
       "analyze":   lambda dt : np.percentile(dt.y,25)
   },
  
    "MF33": {
        "nan2zero":  True,
        "analyze":   lambda dt : np.percentile(dt.X,25)
    },
     
    

    }

   
def int64_to_int(v):
    if type(v) == np.int64 or type(v) == np.int32:
        return int(v)
    else:
        return v

def analyze_Xy_features(X, y, dsd, features, res, pfx="", verbose=False):
    #
    op.TIMER.start()
    for feature_name in features:
       
        if verbose:
            print(f'* ANALYZING FEATURE: {feature_name}')
        #
        if feature_name not in FEATURE_TABLE:
            raise Exception(f'feature "{feature_name}" not in FEATURE_TABLE')
        feature = FEATURE_TABLE[feature_name]
        #
        if "use_numpy" in feature and feature["use_numpy"]:
            X_transform = X.values
        else:
            X_transform = X
            
        if "nan2zero" in feature and feature["nan2zero"]:
            X_transform = nan2zero(X_transform)
        #
        lambda_analyze = FEATURE_TABLE[feature_name]["analyze"]


        # print(f'y(type={type(y)})={y}')
        y = y.to_numpy()
        feature_data = lambda_analyze( Xy_tuple(X=X_transform, y=y) )
        
        if verbose:
            print(f'--> {feature_data}')
        #
        if "flatten" in feature and feature["flatten"]:
            for label, val in zip(feature_data[0], feature_data[1]):
                res[pfx+feature["flatten_pfx"]+label] = int64_to_int(val)
               
        else:
            res[pfx+feature_name] = int64_to_int(feature_data)
        
    op.TIMER.stop()
    total_time = op.TIMER.last_seconds()  
    res['TOTAL_TIME_' + pfx] = total_time


