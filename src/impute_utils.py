#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
from scipy import optimize
from collections import namedtuple

import numpy as np
import pandas as pd

ImputeResult = namedtuple("ImputeResult", ["df", "rmse", "mae", "rows", "cols", "mask_size", "mecha", "impute_time"])


def print_df(df, label=None, max_rows=16):
    if label is not None:
        print(f'+ Dataframe "{label}", TYPE={type(df)}:')
    if type(df) is np.ndarray and max_rows is not None:
        print(df[:max_rows])
        print(f' ... len() = {len(df)} ...')
    elif type(df) is torch.Tensor and max_rows is not None:
        print(df[:max_rows])
        print(f' ... len() = {len(df)} ...')
    else:
        print(df)
    print("")


def color_boolean(val):
    return f'color: {"red" if val else "black"}'

def color_blue(val):
    return f'color: {"red" if val else "black"}'

def generate_html(filename, X_org, X_missing, X_imputed, y_predict=None, y_original=None, label=None):
    if y_predict is not None:
        X_org = pd.concat((X_org, pd.DataFrame(y_predict)), axis=1)
        X_missing = pd.concat((X_missing, pd.DataFrame(y_predict)), axis=1)
        X_imputed = pd.concat((X_imputed, pd.DataFrame(y_predict)), axis=1)
    if y_original is not None:
        X_org = pd.concat((X_org, pd.DataFrame(y_original)), axis=1)
        X_missing = pd.concat((X_missing, pd.DataFrame(y_original)), axis=1)
        X_imputed = pd.concat((X_imputed, pd.DataFrame(y_original)), axis=1)
    mask = np.isnan(X_missing.values)
    mask_df = pd.DataFrame(mask)
    to_style_df = pd.DataFrame(X_imputed.values)
    styled_df = to_style_df.style.apply(lambda _: mask_df.applymap(color_boolean), axis=None).format(precision=2)
    if y_predict is not None:
        styled_df.set_properties(subset=[len(X_imputed.columns)-2], **{'background-color': 'yellow'})
    if y_original is not None:
        styled_df.set_properties(subset=[len(X_imputed.columns)-1], **{'background-color': 'green'})
    with open(filename,'w') as f:
        styled_df.to_html(f)


DFLT_MECHA = "MCAR"
DFLT_OPT = "logistic"
DFLT_P_OBS = 0.5
DFLT_Q = 0.3
DFLT_RANDOM_SEED = None
#############
#2.type
DFLT_MECHA_2 = "MNAR"
DFLT_OPT_2 = "selfmasked"

DFLT_RANDOM_SEED_2 = None
#############

#
# The rest of this file is code by Boris Muzellec:
# Added random seed for torch random
# https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values
# https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
#@inproceedings{muzellec2020missing,
#  title={Missing Data Imputation using Optimal Transport},
#  author={Muzellec, Boris and Josse, Julie and Boyer, Claire and Cuturi, Marco},
#  booktitle={International Conference on Machine Learning},
#  pages={7130--7140},
#  year={2020},
#  organization={PMLR}
#}

def produce_NA(X, p_miss, mecha=DFLT_MECHA, random_seed=DFLT_RANDOM_SEED, opt=DFLT_OPT, p_obs=DFLT_P_OBS, q=DFLT_Q):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    if random_seed is not None:
        torch.manual_seed(random_seed) # for MCAR
        np.random.seed(random_seed) # for MNAR
            
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    elif mecha == "MCAR":
        mask = (torch.rand(X.shape) < p_miss).double()
    else:
        raise Exception(f'produce_NA(): unknown settings for missing values {mecha}')
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    if False:
        print(f'MASK = {mask}')
        print(f'TRY = {np.isnan(X_nas.double().numpy()).astype(bool)}')
        sys.exit(0)
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}
############################
def produce_NA_2(X, p_miss, mecha=DFLT_MECHA, random_seed=DFLT_RANDOM_SEED, opt=DFLT_OPT, p_obs=DFLT_P_OBS, q=DFLT_Q):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    if random_seed is not None:
        torch.manual_seed(random_seed) # for MCAR
        np.random.seed(random_seed) # for MNAR
            
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    elif mecha == "MCAR":
        mask = (torch.rand(X.shape) < p_miss).double()
    else:
        raise Exception(f'produce_NA(): unknown settings for missing values {mecha}')
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    if False:
        print(f'MASK = {mask}')
        print(f'TRY = {np.isnan(X_nas.double().numpy()).astype(bool)}')
        sys.exit(0)
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}

############################

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()


def ACCURACY(X, X_true, mask):
    if torch.is_tensor(mask):
        raise Exception("INCOMPLETE")
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        acc = (X[mask_] == X_true[mask_]).sum() / mask_.sum()
        if False:
            print(f'*** INCOMPLETE, check ACCURACY IMPLEMENTATION')
            print(f'*** mask_ =\n{mask_}')
            print(f'*** X     =\n{X}')
            print(f'*** X_true=\n{X_true}')
            print(f'*** shape={mask_.shape}')
            print(f'*** mask_count={mask_.sum()}')
            print(f'*** true_count={(X[mask_] == X_true[mask_]).sum()}')
            print(f'*** acc={acc}')
        return acc

def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        res = np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())
        return res


##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask

##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask

def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


VERBOSE = True
ROWS = 10
COLS = 3
P_MISS = 0.5
RANDOM_SEED = None
REPEAT = 2


if __name__ == "__main__":
    if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)
    T = torch.rand(ROWS, COLS)
    N = ROWS * COLS
    if VERBOSE:
        print(f'#')
        print(f'# Testing Boris Muzellecs missings values algorithms')
        print(f'#')
        print(f'!P_MISS = "{P_MISS}"')
        print(f'!MECHA  = "{DFLT_MECHA}"')
        print(f'!OPT    = "{DFLT_OPT}"')
        print(f'!P_OBS  = "{DFLT_P_OBS}"')
        print(f'!P_OBS  = "{DFLT_P_OBS}"')
        print(f'!Q      = "{DFLT_Q}"')
        print(f'!ROWS   = "{ROWS}"')
        print(f'!COLS   = "{COLS}"')
        print(f'!REPEAT = "{REPEAT}"')
        Nan_sum = 0
    if VERBOSE:
        print(f'\n!Generated random tensor\n{T}')
    for i in range(REPEAT):
        R = produce_NA(T, P_MISS)
        T_res = R['X_incomp']
        if VERBOSE:
            print(f'\n!Result Tensor by produce_NA()\n{T_res}')
            N_nan = torch.isnan(T_res.view(-1)).sum()
            print(f'nan_count = {N_nan}/{N} == {N_nan/N:1.4}%')
            Nan_sum += N_nan
    if VERBOSE:
        print(f'\nTOTAL PERCENTAGE Nan: {Nan_sum/(N*REPEAT):1.4}%')
