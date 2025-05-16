#For VAE and GAIN imputer we utilized this paper's code and settings.
#@ARTICLE{imputation_benchmark_jaeger_2021,
#	AUTHOR={Jäger, Sebastian and Allhorn, Arndt and Bießmann, Felix},
#	TITLE={A Benchmark for Data Imputation Methods},
#	JOURNAL={Frontiers in Big Data},
#	VOLUME={4},
#	PAGES={48},
#	YEAR={2021},
#	URL={https://www.frontiersin.org/article/10.3389/fdata.2021.693674},
#	DOI={10.3389/fdata.2021.693674},
#	ISSN={2624-909X},
#	ABSTRACT={With the increasing importance and complexity of data pipelines, data quality became one of the key challenges in modern software applications. The importance of data quality has been recognized beyond the field of data engineering and database management systems (DBMSs). Also, for machine learning (ML) applications, high data quality standards are crucial to ensure robust predictive performance and responsible usage of automated decision making. One of the most frequent data quality problems is missing values. Incomplete datasets can break data pipelines and can have a devastating impact on downstream ML applications when not detected. While statisticians and, more recently, ML researchers have introduced a variety of approaches to impute missing values, comprehensive benchmarks comparing classical and modern imputation approaches under fair and realistic conditions are underrepresented. Here, we aim to fill this gap. We conduct a comprehensive suite of experiments on a large number of datasets with heterogeneous data and realistic missingness conditions, comparing both novel deep learning approaches and classical ML imputation methods when either only test or train and test data are affected by missing data. Each imputation method is evaluated regarding the imputation quality and the impact imputation has on a downstream ML task. Our results provide valuable insights into the performance of a variety of imputation methods under realistic conditions. We hope that our results help researchers and engineers to guide their data preprocessing method selection for automated data quality improvement.}
#}


from jager_vae import VAEImputer, GAINImputer
#
import operations as op
from dataset_loader import get_dataset
from impute_utils import RMSE, MAE
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import initializers


IMPUTER   = 'VAE'
# IMPUTER   = 'GAIN'
# DATASET   = "Wisconsin"
DATASET   = "bupa1"
NORMALIZE = True
MISSING   = 0.7
MECHA     = 'MCAR'
SEED      = 1002

dsd, df = get_dataset(DATASET, original=True)

X = df[dsd.feature_columns()]
# y = df[dsd.label_column()].astype(np.float64)

if NORMALIZE:
    scaler = StandardScaler()
    X[dsd.numeric_columns()] = scaler.fit_transform(X[dsd.numeric_columns()])
    #
    label_encoder = LabelEncoder()
    X_new = X.copy()
    for ccol in dsd.catagorical_columns():
        X_new[ccol] = label_encoder.fit_transform(X[ccol])
    X = X_new

X_missing = op.generate_missing(X, MISSING, MECHA, SEED)

if IMPUTER == 'VAE':
    imputer = VAEImputer()
elif IMPUTER == 'GAIN':
    imputer = GAINImputer()
else:
    raise Exception(f'unknow imputer: "{IMPUTER}"')

if True:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    #
    initializers_list = [
        initializers.RandomNormal,
        initializers.RandomUniform,
        initializers.TruncatedNormal,
        initializers.VarianceScaling,
        initializers.GlorotNormal,
        initializers.GlorotUniform,
        initializers.HeNormal,
        initializers.HeUniform,
        initializers.LecunNormal,
        initializers.LecunUniform,
        initializers.Orthogonal,
    ]
    for initializer in initializers_list:
        print(f"Running {initializer}")

        for iteration in range(2):
            # In order to get same results across multiple runs from an initializer,
            # you can specify a seed value.
            result = float(initializer(seed=SEED)(shape=(1, 1)))
            print(f"\tIteration --> {iteration} // Result --> {result}")
        print("\n")


imputer.fit(X_missing, target_columns=list(X_missing.columns))

X_imputed, _ = imputer.transform(X_missing)

mask = np.isnan(X_missing.values)
# mask = np.logical_not(mask)

rmse = RMSE(X_imputed.values, X.values, mask)
mae  =  MAE(X_imputed.values, X.values, mask)

print(f'X =\n{X}\n')
print(f'X_missing =\n{X_missing}\n')
print(f'X_imputed =\n{X_imputed}\n')

print(f'* Running {IMPUTER} imputer:')
print(f'+ DATASET   = {DATASET}')
print(f'+ NORMALIZE = {NORMALIZE}')
print(f'+ MISSING   = {MISSING}')
print(f'+ MECHA     = {MECHA}')
print(f'+ SEED      = {SEED}')
print(f'= RMSE      = {rmse}')
print(f'= MAE       = {mae}')

# print(f'X_imputed({type(X_imputed)}) =\n{X_imputed}')
