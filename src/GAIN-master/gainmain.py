# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:20:06 2023


"""

from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split
from gain import gain
from utils import rmse_loss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import rmse_loss
from sklearn.preprocessing import MinMaxScaler
data = load_breast_cancer()
x = data.data
y = data.target
y = y.reshape(-1, 1)
scaler = MinMaxScaler()


x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,  test_size=0.2, random_state=42
)
missing_percentage = 0.1 
missing_mask = np.random.rand(*x_train.shape) < missing_percentage
mask_train = np.where(missing_mask, np.nan, x_train)

x_missing = np.where(missing_mask, np.nan, x_train)




def rmse_loss(imputed_data, original_data, data_m=None):
    if data_m is None:
       data_m = np.ones_like(original_data)  
    diff = (original_data - imputed_data) * data_m
    mse = np.sum(np.square(diff)) / np.sum(data_m)
    rmse = np.sqrt(mse)
    return rmse


# GAIN parametreleri
GAIN_PARAMETERS = {'batch_size': 128,
                   'hint_rate': 0.1,
                   'alpha': 100,
                   'iterations': 10000}


imputed_data = gain(x_train, GAIN_PARAMETERS)

ori_data_x = x_train  

loss = rmse_loss(imputed_data, x_train)


print(f"RMSE Loss: {loss}")


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Real")
plt.imshow(x_train, aspect="auto", cmap="viridis")

plt.subplot(1, 2, 2)
plt.title("Completed")
plt.imshow(imputed_data, aspect="auto", cmap="viridis")

plt.show()
print(imputed_data.shape)
print(imputed_data)
print(x_train.shape)
print(x_train)
loss = rmse_loss(imputed_data, x_train, mask_train)
