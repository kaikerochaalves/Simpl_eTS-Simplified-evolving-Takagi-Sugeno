# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Kaike Alves
"""

# Import libraries
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import statistics as st
from dieboldmariano import dm_test

# Neural Network
# Model
from keras.models import Sequential
# Layers
from keras.layers import InputLayer, Dense, Dropout, Conv1D, GRU, LSTM, MaxPooling1D, Flatten, SimpleRNN
# Optimizers
from keras.optimizers import SGD, Adam
# Save network
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys

# Models
sys.path.append(r'Models')
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lssvr import LSSVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from eTS import eTS
from Simpl_eTS import Simpl_eTS
from exTS import exTS
from ePL import ePL
from eMG import eMG
from ePL_plus import ePL_plus
from ePL_KRLS_DISCO import ePL_KRLS_DISCO
from NFISiS import NTSK, NewMamdaniRegressor
from GEN_NFISiS import GEN_NTSK, GEN_NMR
from R_NFISiS import R_NTSK, R_NMR


# Import the series
import yfinance as yf


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "SP500"

horizon = 5
    
# Importing the data
SP500 = yf.Ticker('^GSPC')
SP500 = SP500.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = SP500.columns
Data = SP500[columns[:4]]

# Add the target column value
NextClose = Data.iloc[horizon:,-1].values
Data = Data.drop(Data.index[-horizon:])
Data['NextClose'] = NextClose

# Convert to array
X = Data[Data.columns[:-1]].values
y = Data[Data.columns[-1]].values

# Spliting the data into train, validation, and test
n = Data.shape[0]
training_size = round(n*0.6)
validation_size = round(n*0.8)
X_train, X_val, X_test = X[:training_size,:], X[training_size:validation_size,:], X[validation_size:,:]
y_train, y_val, y_test = y[:training_size], y[training_size:validation_size:], y[validation_size:]

# Fixing y shape
y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Min-max scaling X
scaler = MinMaxScaler()
X = scaler.fit_transform(X, y)
X_DL = X.reshape(X.shape[0], X.shape[1], 1)

# Reshape the inputs for deep learning
X_train_DL = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_DL = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_DL = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the expanding window data
expanding_window = TimeSeriesSplit(n_splits=10)



#-----------------------------------------------------------------------------
# Initialize dataframe
#-----------------------------------------------------------------------------

# Summary
columns = ["Model_Name", "NRMSE", "NDEI", "MAPE", "CPPM", "Rules", "Best_Params", "NRMSE_l", "NDEI_l", "MAPE_l", "CPPM_l", "Rules_l", "Summary_Results"]
results = pd.DataFrame(columns = columns)

y_test_exp = np.array([])
for train, test in expanding_window.split(X):
    
    # Split train and test
    y_test_part = y[test]
    y_test_exp = np.append(y_test_exp, y_test_part.flatten())

# Predictions
predictions = pd.DataFrame()
predictions['y_test'] = y_test_exp

#-----------------------------------------------------------------------------
# Classic Models
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# KNN
#-----------------------------------------------------------------------------

Model_Name = r"KNN \cite{fix1951discriminatory}"

# Define Grid Search parameters
parameters = {'n_neighbors': [2, 3, 5, 11]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    KNN = KNeighborsRegressor(**param)
    KNN.fit(X_train,y_train)
    # Make predictions
    y_pred = KNN.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_KNN_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    KNN = KNeighborsRegressor(**best_KNN_params)
    KNN.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = KNN.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    KNN_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_KNN_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, KNN_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    KNN_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_KNN_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", KNN_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(KNN_)

#-----------------------------------------------------------------------------
# Regression Tree
#-----------------------------------------------------------------------------

Model_Name = r"Regression Tree \cite{breiman1984classification}"

# Define Grid Search parameters
parameters = {'max_depth': [2, 4, 8, 12, 16, 20],'max_features': [2, 4, 6, 8, 10]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    RT = DecisionTreeRegressor(**param)
    RT.fit(X_train,y_train)
    # Make predictions
    y_pred = RT.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_RT_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    RT = DecisionTreeRegressor(**best_RT_params)
    RT.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = RT.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    RT_ = f'{Model_Name[:15]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_RT_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RT_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    RT_ = f'{Model_Name[:15]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_RT_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", RT_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(RT_)

#-----------------------------------------------------------------------------
# Random Forest
#-----------------------------------------------------------------------------

Model_Name = r"Random Forest \cite{ho1995random}"

# Define Grid Search parameters
parameters = {'n_estimators': [50, 100, 150, 200],'max_depth': [2, 4, 8, 12, 16, 20],'max_features': [2, 4, 6, 8, 10]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    RF = RandomForestRegressor(**param)
    RF.fit(X_train,y_train)
    # Make predictions
    y_pred = RF.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_RF_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    RF = RandomForestRegressor(**best_RF_params)
    RF.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = RF.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

# Store outputs and error
y_pred_RF = y_pred
RMSE_RF = RMSE

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    RF_ = f'{Model_Name[:13]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_RF_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RF_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    RF_ = f'{Model_Name[:13]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_RF_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", RF_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(RF_)

#-----------------------------------------------------------------------------
# SVM
#-----------------------------------------------------------------------------

Model_Name = r"SVM \cite{cortes1995support}"

# Define Grid Search parameters
parameters1 = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C':[0.01,0.1,1,10], 'gamma': [0.01,0.5,1,10,50]}
grid1 = ParameterGrid(parameters1)

higher_cppm = -np.inf
for param in grid1:
    
    #print(param)
    
    # Optimize parameters
    SVM = SVR(**param)
    SVM.fit(X_train,y_train)
    # Make predictions
    y_pred = SVM.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_SVM_params = param

# # Use best parameters1 to save time looking for best parameters2
# parameters2 = {'kernel': ['poly'], 'C':[best_SVM_params['C']], 'gamma': [best_SVM_params['gamma']], 'degree': [2,3]}

# grid2 = ParameterGrid(parameters2)

# for param in grid2:
    
#     #print(param)
    
#     # Optimize parameters
#     SVM = SVR(**param)
#     SVM.fit(X_train,y_train)
#     # Make predictions
#     y_pred = SVM.predict(X_val)
#     
    
#     # Calculating the error metrics
#     # Compute the Root Mean Square Error
#     RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
#     if RMSE < lower_rmse:
#         lower_rmse = RMSE
#         best_SVM_params = param


# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    SVM = SVR(**best_SVM_params)
    SVM.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = SVM.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    SVM_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_SVM_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, SVM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    SVM_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_SVM_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", SVM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(SVM_)

#-----------------------------------------------------------------------------
# LS-SVM
#-----------------------------------------------------------------------------

Model_Name = r"LS-SVM \cite{suykens1999least}"

# Define Grid Search parameters
parameters = {'kernel': ['linear']}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid1:
    
    #print(param)
    
    # Optimize parameters
    LS_SVR = LSSVR(**param)
    LS_SVR.fit(X_train,y_train)
    # Make predictions
    y_pred = LS_SVR.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_LS_SVR_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])

# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    LS_SVR = LSSVR(**best_LS_SVR_params)
    LS_SVR.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = LS_SVR.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    LS_SVR_ = f'{Model_Name[:6]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_LS_SVR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, LS_SVR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    LS_SVR_ = f'{Model_Name[:6]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_LS_SVR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", LS_SVR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(LS_SVR_)

#-----------------------------------------------------------------------------
# Gradient Boosting
#-----------------------------------------------------------------------------

Model_Name = r"GBM \cite{friedman2001greedy}"

# Define Grid Search parameters
parameters = {'learning_rate':[0.01, 0.05, 0.1], 'n_estimators': [50, 100, 150, 200],'max_depth': [2, 4, 8, 12, 16, 20],'max_features': [2, 4, 6, 8, 10]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    GBM = GradientBoostingRegressor(**param)
    GBM.fit(X_train,y_train)
    # Make predictions
    y_pred = GBM.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_GBM_params = param
        
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    GBM = GradientBoostingRegressor(**best_GBM_params)
    GBM.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = GBM.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    GBM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GBM_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GBM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GBM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GBM_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GBM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
print(GBM_)

# -----------------------------------------------------------------------------
# XGBoost
# -----------------------------------------------------------------------------

Model_Name = r"XGBoost \cite{chen2016xgboost}"

# Define Grid Search parameters
parameters = {'n_estimators':[50, 100, 150, 200], 'min_child_weight':[2,5], 
              'gamma':[i/10.0 for i in range(3,6)],'max_depth': [2, 4, 8, 12, 16, 20],
              'eval_metric':['rmse'],'eta':[i/10.0 for i in range(3,6)]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    XGBoost = XGBRegressor(**param)
    XGBoost.fit(X_train,y_train)
    # Make predictions
    y_pred = XGBoost.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_XGBoost_params = param


# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    XGBoost = XGBRegressor(**best_XGBoost_params)
    XGBoost.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = XGBoost.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    XGBoost_ = f'{Model_Name[:7]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_XGBoost_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, XGBoost_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    XGBoost_ = f'{Model_Name[:7]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_XGBoost_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", XGBoost_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
print(XGBoost_)

#-----------------------------------------------------------------------------
# Light-GBM Regressor
#-----------------------------------------------------------------------------

Model_Name = r"LGBM \cite{ke2017lightgbm}"

# Define Grid Search parameters
parameters = {'n_estimators':[50, 100, 150, 200], 'learning_rate':[0.01, 0.05, 0.1, 0.5],'max_depth': [2, 4, 8, 12, 16, 20],'max_features': [2, 4, 6, 8, 10],'verbosity':[-1]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    LGBM = LGBMRegressor(**param)
    LGBM.fit(X_train,y_train)
    # Make predictions
    y_pred = LGBM.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_LGBM_params = param
        
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    LGBM = LGBMRegressor(**best_LGBM_params)
    LGBM.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = LGBM.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    LGBM_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_LGBM_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, LGBM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    LGBM_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_LGBM_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", LGBM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
print(LGBM_)

#-----------------------------------------------------------------------------
# Deep Learning
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# MLP
#-----------------------------------------------------------------------------


Model_Name = r"MLP \cite{rosenblatt1958perceptron}"

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, activation = "relu", learning_rate=3e-3, input_shape=[4]):
    model = Sequential()
    model.add(InputLayer(shape=(input_shape,)))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_MLP_parameters = {'n_hidden':0, 'n_neurons':0, 'activation':"relu", 'learning_rate': 0.067, 'input_shape': X_train_DL.shape[1]}

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
part = 0
for train, test in expanding_window.split(X_DL):
    
    part += 1
    
    # Split train and test
    X_train_part, y_train_part = X_DL[train], y[train]
    X_test_part, y_test_part = X_DL[test], y[test]
    
    # Checkpoint for early stopping
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name[:3]}_part{part}_horizon{horizon}.keras', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

    # Define the model
    model = build_model(**best_MLP_parameters)

    # Fit the model
    history = model.fit(X_train_part, y_train_part, epochs = 1000, validation_data=(X_test_part, y_test_part), callbacks=[checkpoint_cb, early_stopping_cb])
    
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    MLP_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_MLP_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, MLP_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    MLP_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_MLP_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", MLP_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(MLP_)

#-----------------------------------------------------------------------------
# CNN
#-----------------------------------------------------------------------------


Model_Name = r"CNN \cite{fukushima1980neocognitron}"

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_CNN_parameters = {'n_hidden':3, 'n_neurons':99, 'learning_rate': 0.007}

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
part = 0
for train, test in expanding_window.split(X_DL):
    
    part += 1
    
    # Split train and test
    X_train_part, y_train_part = X_DL[train], y[train]
    X_test_part, y_test_part = X_DL[test], y[test]
    
    # Checkpoint for early stopping
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name[:3]}_part{part}_horizon{horizon}.keras', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

    # Define the model
    model = build_model(**best_CNN_parameters)

    # Fit the model
    history = model.fit(X_train_part, y_train_part, epochs = 1000, validation_data=(X_test_part, y_test_part), callbacks=[checkpoint_cb, early_stopping_cb])
    
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    CNN_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_CNN_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, CNN_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    CNN_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_CNN_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", CNN_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(CNN_)

#-----------------------------------------------------------------------------
# RNN
#-----------------------------------------------------------------------------

Model_Name = r"RNN \cite{hopfield1982neural}"

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3):
    model = Sequential()
    if n_hidden == 1:
        model.add(SimpleRNN(n_neurons, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    elif n_hidden == 2:
        model.add(SimpleRNN(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        model.add(SimpleRNN(n_neurons))
    else:
        model.add(SimpleRNN(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        for layer in range(1, n_hidden-1):
            model.add(SimpleRNN(n_neurons, return_sequences=True))
        model.add(SimpleRNN(n_neurons))
    
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_RNN_parameters = {'n_hidden':1, 'n_neurons':80, 'learning_rate': 0.190}

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
part = 0
for train, test in expanding_window.split(X_DL):
    
    part += 1
    
    # Split train and test
    X_train_part, y_train_part = X_DL[train], y[train]
    X_test_part, y_test_part = X_DL[test], y[test]
    
    # Checkpoint for early stopping
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name[:3]}_part{part}_horizon{horizon}.keras', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

    # Define the model
    model = build_model(**best_RNN_parameters)

    # Fit the model
    history = model.fit(X_train_part, y_train_part, epochs = 1000, validation_data=(X_test_part, y_test_part), callbacks=[checkpoint_cb, early_stopping_cb])
    
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    RNN_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_RNN_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RNN_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    RNN_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_RNN_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", RNN_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
print(RNN_)
 
#-----------------------------------------------------------------------------
# LSTM
#-----------------------------------------------------------------------------

Model_Name = r"LSTM \cite{hochreiter1997long}"

# Define the function to create models for the optimization method
def build_model(n_neurons=30, n_lstm_hidden = 1, neurons_dense = 30, dropout_rate = 0, n_dense_hidden = 1, learning_rate=3e-3):
    model = Sequential()
    if n_lstm_hidden == 1:
        model.add(LSTM(n_neurons, return_sequences=False, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    elif n_lstm_hidden == 2:
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        model.add(LSTM(n_neurons, return_sequences=False))
    else:
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        for layer in range(1, n_lstm_hidden-1):
            model.add(LSTM(n_neurons, return_sequences=True))
        model.add(LSTM(n_neurons, return_sequences=False))
    for dense_layer in range(n_dense_hidden):
        model.add(Dense(neurons_dense))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_LSTM_parameters = {'n_neurons':72, 'n_lstm_hidden':5, 'neurons_dense': 1, 'dropout_rate': 0, 'n_dense_hidden':3, 'learning_rate': 0.120}

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
part = 0
for train, test in expanding_window.split(X_DL):
    
    part += 1
    
    # Split train and test
    X_train_part, y_train_part = X_DL[train], y[train]
    X_test_part, y_test_part = X_DL[test], y[test]
    
    # Checkpoint for early stopping
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name[:4]}_part{part}_horizon{horizon}.keras', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

    # Define the model
    model = build_model(**best_LSTM_parameters)

    # Fit the model
    history = model.fit(X_train_part, y_train_part, epochs = 1000, validation_data=(X_test_part, y_test_part), callbacks=[checkpoint_cb, early_stopping_cb])
    
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    LSTM_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_LSTM_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, LSTM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    LSTM_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_LSTM_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", LSTM_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(LSTM_)

#-----------------------------------------------------------------------------
# GRU
#-----------------------------------------------------------------------------

Model_Name = r"GRU \cite{chung2014empirical}"

# Define the function to create models for the optimization method
def build_model(filters = 64, kernel_size = 2, strides = 1, n_neurons=30, n_gru_hidden = 1, neurons_dense = 30, dropout_rate = 0, n_dense_hidden = 1, learning_rate=3e-3):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="valid", input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    if n_gru_hidden == 1:
        model.add(GRU(n_neurons, return_sequences=False))
    elif n_gru_hidden == 2:
        model.add(GRU(n_neurons, return_sequences=True))
        model.add(GRU(n_neurons, return_sequences=False))
    else:
        #model.add(keras.layers.GRU(n_neurons, return_sequences=True))
        for layer in range(n_gru_hidden-1):
            model.add(GRU(n_neurons, return_sequences=True))
        model.add(GRU(n_neurons, return_sequences=False))
    for dense_layer in range(n_dense_hidden):
        model.add(Dense(neurons_dense))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_GRU_parameters = {'filters':8, 'kernel_size':1, 'strides': 5, 'n_neurons': 58, 'n_gru_hidden':2, 'neurons_dense': 1, 'dropout_rate': 0, 'n_dense_hidden': 4, 'learning_rate': 0.009}

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
part = 0
for train, test in expanding_window.split(X_DL):
    
    part += 1
    
    # Split train and test
    X_train_part, y_train_part = X_DL[train], y[train]
    X_test_part, y_test_part = X_DL[test], y[test]
    
    # Checkpoint for early stopping
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name[:3]}_part{part}_horizon{horizon}.keras', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

    # Define the model
    model = build_model(**best_GRU_parameters)

    # Fit the model
    history = model.fit(X_train_part, y_train_part, epochs = 1000, validation_data=(X_test_part, y_test_part), callbacks=[checkpoint_cb, early_stopping_cb])
    
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    GRU_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GRU_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GRU_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GRU_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GRU_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GRU_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
print(GRU_)

#-----------------------------------------------------------------------------
# WaveNet
#-----------------------------------------------------------------------------

Model_Name = r"WaveNet \cite{oord2016wavenet}"

# Define the function to create models for the optimization method
def build_model(dilation_rate=(1, 2, 4, 8), repeat=2, learning_rate=3e-3):
    model = Sequential()
    model.add(InputLayer(input_shape=[X_train_DL.shape[1],X_train_DL.shape[2]]))
    for rate in dilation_rate * repeat:
        model.add(Conv1D(filters=20, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
    model.add(Conv1D(filters=10, kernel_size=1))
    model.add(Flatten())
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_WaveNet_parameters = {'dilation_rate':(1,2,4,8,16), 'repeat':2, 'learning_rate': 0.003}

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
part = 0
for train, test in expanding_window.split(X_DL):
    
    part += 1
    
    # Split train and test
    X_train_part, y_train_part = X_DL[train], y[train]
    X_test_part, y_test_part = X_DL[test], y[test]
    
    # Checkpoint for early stopping
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name[:7]}_part{part}_horizon{horizon}.keras', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

    # Define the model
    model = build_model(**best_WaveNet_parameters)

    # Fit the model
    history = model.fit(X_train_part, y_train_part, epochs = 1000, validation_data=(X_test_part, y_test_part), callbacks=[checkpoint_cb, early_stopping_cb])
    
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    Rules = "-"
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    WaveNet_ = f'{Model_Name[:7]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_WaveNet_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, WaveNet_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    WaveNet_ = f'{Model_Name[:7]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_WaveNet_parameters.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", WaveNet_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(WaveNet_)

#-----------------------------------------------------------------------------
# evolving Fuzzy Systems
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# eTS
#-----------------------------------------------------------------------------


Model_Name = r"eTS \cite{angelov2004approach}"

# Define Grid Search parameters
parameters = {'omega': [50, 100, 250, 500, 1000, 10000], 'r': [0.1, 0.3, 0.5, 0.7, 5, 10, 50]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = eTS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_eTS_params = param
        
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = eTS(**best_eTS_params)
    OutputTraining, Rules = model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules[-1])
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    eTS_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_eTS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, eTS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    eTS_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_eTS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", eTS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(eTS_)

#-----------------------------------------------------------------------------
# Simpl_eTS
#-----------------------------------------------------------------------------


Model_Name = r"Simpl\_eTS \cite{angelov2005simpl_ets}"

# Define Grid Search parameters
parameters = {'omega': [50, 250, 500, 750, 1000], 'r': [0.1, 0.3, 0.5, 0.7]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = Simpl_eTS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_Simpl_eTS_params = param
        
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = Simpl_eTS(**best_Simpl_eTS_params)
    OutputTraining, Rules = model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules[-1])
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    Simpl_eTS_ = f'{Model_Name[:10]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_Simpl_eTS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, Simpl_eTS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    Simpl_eTS_ = f'{Model_Name[:10]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_Simpl_eTS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", Simpl_eTS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(Simpl_eTS_)

#-----------------------------------------------------------------------------
# exTS
#-----------------------------------------------------------------------------


Model_Name = r"exTS \cite{angelov2006evolving}"

# Define Grid Search parameters
parameters = {'omega': [50, 250, 750, 1000], 'mu': [0.1, 0.3, 0.5, 0.7]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = exTS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_exTS_params = param
    
    
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = exTS(**best_exTS_params)
    OutputTraining, Rules = model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules[-1])
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    exTS_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_exTS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, exTS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    exTS_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_exTS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", exTS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(exTS_)

#-----------------------------------------------------------------------------
# ePL
#-----------------------------------------------------------------------------


Model_Name = r"ePL \cite{lima2010evolving}"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01, 0.1, 0.9], 'beta': [0.001, 0.005, 0.01, 0.1, 0.2], 'lambda1': [0.001, 0.1], 's': [100, 10000], 'r': [0.1, 0.25, 0.5, 0.75]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = ePL(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_ePL_params = param
        
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = ePL(**best_ePL_params)
    OutputTraining, Rules = model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules[-1])
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    ePL_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_ePL_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, ePL_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    ePL_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_ePL_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", ePL_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(ePL_)

#-----------------------------------------------------------------------------
# eMG
#-----------------------------------------------------------------------------


Model_Name = r"eMG \cite{lemos2010multivariable}"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01], 'lambda1': [0.1, 0.5], 'w': [10, 50], 'sigma': [0.001, 0.003], 'omega': [10**4]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = eMG(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_eMG_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = eMG(**best_eMG_params)
    OutputTraining, Rules = model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules[-1])
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    eMG_ = f'{Model_Name[:3]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_eMG_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, eMG_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    eMG_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_eMG_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", eMG_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(eMG_)

#-----------------------------------------------------------------------------
# ePL+
#-----------------------------------------------------------------------------


Model_Name = r"ePL+ \cite{maciel2012enhanced}"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01, 0.1], 'beta': [0.01, 0.1, 0.25], 'lambda1': [0.25, 0.5, 0.75], 'omega': [100, 10000], 'sigma': [0.1, 0.25, 0.5], 'e_utility': [0.03, 0.05], 'pi': [0.3, 0.5]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = ePL_plus(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_ePL_plus_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = ePL_plus(**best_ePL_plus_params)
    OutputTraining, Rules = model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules[-1])
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    ePL_plus_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_ePL_plus_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, ePL_plus_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    ePL_plus_ = f'{Model_Name[:4]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_ePL_plus_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", ePL_plus_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(ePL_plus_)

#-----------------------------------------------------------------------------
# ePL-KRLS-DISCO
#-----------------------------------------------------------------------------


Model_Name = r"ePL-KRLS-DISCO \cite{alves2021novel}"

# Define Grid Search parameters
parameters = {'alpha': [0.05, 0.1], 'beta': [0.01, 0.1, 0.25], 'lambda1': [1e-7, 0.001], 'sigma': [0.5, 1, 10, 50], 'e_utility': [0.03, 0.05]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = ePL_KRLS_DISCO(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_ePL_KRLS_DISCO_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = ePL_KRLS_DISCO(**best_ePL_KRLS_DISCO_params)
    OutputTraining, Rules = model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules[-1])
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    ePL_KRLS_DISCO_ = f'{Model_Name[:14]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_ePL_KRLS_DISCO_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, ePL_KRLS_DISCO_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    ePL_KRLS_DISCO_ = f'{Model_Name[:14]} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_ePL_KRLS_DISCO_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", ePL_KRLS_DISCO_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(ePL_KRLS_DISCO_)

#-----------------------------------------------------------------------------
# Proposed Models
#-----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# NMR
# -----------------------------------------------------------------------------


Model_Name = "NMR"

# Set hyperparameters range
parameters = {'rules':range(1,20), 'fuzzy_operator':["prod","min","max","minmax"]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = NewMamdaniRegressor(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_NMR_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = NewMamdaniRegressor(**best_NMR_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = model.parameters.shape[0]
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    NMR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NMR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, NMR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    NMR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NMR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", NMR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(NMR_)

# -----------------------------------------------------------------------------
# NTSK-RLS
# -----------------------------------------------------------------------------

Model_Name = "NTSK-RLS"

# Set hyperparameters range
parameters = {'rules':[1], 'lambda1':[0.95,0.96,0.97,0.98,0.99], 'adaptive_filter':["RLS"], 'fuzzy_operator':["prod","min","max","minmax"]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_NTSK_RLS_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = NTSK(**best_NTSK_RLS_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = model.parameters.shape[0]
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    NTSK_RLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_RLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, NTSK_RLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    NTSK_RLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_RLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", NTSK_RLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(NTSK_RLS_)

# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------

Model_Name = "NTSK-wRLS"

# Set hyperparameters range
parameters = {'rules':range(1,20), 'adaptive_filter':["wRLS"], 'fuzzy_operator':["prod","min","max","minmax"]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_NTSK_wRLS_params = param
    
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = NTSK(**best_NTSK_wRLS_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = model.parameters.shape[0]
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    NTSK_wRLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_wRLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, NTSK_wRLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    NTSK_wRLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_wRLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", NTSK_wRLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(NTSK_wRLS_)

# -----------------------------------------------------------------------------
# GEN-NMR
# -----------------------------------------------------------------------------


Model_Name = "GEN-NMR"

# Set hyperparameters range
parameters = {'rules':range(1,20, 2), 'fuzzy_operator':["prod","min","max","minmax"],
              'num_generations':[10], 'num_parents_mating':[5], 'sol_per_pop':[10], 'error_metric':["RMSE","MAE","CPPM"], 'parallel_processing':[10]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NMR(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_GEN_NMR_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = GEN_NMR(**best_GEN_NMR_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = model.parameters.shape[0]
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    GEN_NMR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GEN_NMR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GEN_NMR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GEN_NMR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GEN_NMR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GEN_NMR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(GEN_NMR_)

# -----------------------------------------------------------------------------
# GEN-NTSK-RLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-RLS"

# Set hyperparameters range
parameters = {'rules':[1], 'lambda1':[0.95,0.96,0.97,0.98,0.99], 'adaptive_filter':["RLS"], 'fuzzy_operator':["prod","min","max","minmax"],
              'num_generations':[5], 'num_parents_mating':[5], 'sol_per_pop':[5], 'error_metric':["RMSE","MAE","CPPM"], 'parallel_processing':[10]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_GEN_NTSK_RLS_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = GEN_NTSK(**best_GEN_NTSK_RLS_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = model.parameters.shape[0]
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    GEN_NTSK_RLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GEN_NTSK_RLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GEN_NTSK_RLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GEN_NTSK_RLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_GEN_NTSK_RLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GEN_NTSK_RLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(GEN_NTSK_RLS_)

# -----------------------------------------------------------------------------
# GEN-NTSK-wRLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-wRLS"

# Set hyperparameters range
parameters = {'rules':range(1,20,2), 'adaptive_filter':["wRLS"], 'fuzzy_operator':["prod","min","max","minmax"],
              'num_generations':[5], 'num_parents_mating':[5], 'sol_per_pop':[5], 'error_metric':["RMSE","MAE","CPPM"], 'parallel_processing':[10]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_GEN_NTSK_wRLS_params = param
    
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = GEN_NTSK(**best_GEN_NTSK_wRLS_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = model.parameters.shape[0]
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    GEN_NTSK_wRLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_wRLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GEN_NTSK_wRLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GEN_NTSK_wRLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_wRLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GEN_NTSK_wRLS_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(GEN_NTSK_wRLS_)

# -----------------------------------------------------------------------------
# R-NMR
# -----------------------------------------------------------------------------


Model_Name = "R-NMR"

# Set hyperparameters range
parameters = {'n_estimators':[50], 'combination':["mean","median","weighted_average"]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = R_NMR(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_R_NMR_params = param

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = R_NMR(**best_R_NMR_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = "-"
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    R_NMR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_R_NMR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, R_NMR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    R_NMR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_R_NMR_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", R_NMR_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(R_NMR_)

# -----------------------------------------------------------------------------
# R-NTSK
# -----------------------------------------------------------------------------


Model_Name = "R-NTSK"

# Set hyperparameters range
parameters = {'n_estimators':[50], 'combination':["mean","median","weighted_average"]}
grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = R_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_val[1:]
    current_y = y_val[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred[1:]
    current_y_pred = y_pred[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    
    if CPPM > higher_cppm:
        higher_cppm = CPPM
        best_R_NTSK_params = param
    
# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])
Rules_l = np.array([])
# Perform simulations with expanding window
for train, test in expanding_window.split(X):
    
    # Split train and test
    X_train_part, y_train_part = X[train], y[train]
    X_test_part, y_test_part = X[test], y[test]
    
    # Optimized parameters
    model = R_NTSK(**best_R_NTSK_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = "-"
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

# Store outputs and error
y_pred_R_NTSK = y_pred
RMSE_R_NTSK = RMSE

if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    R_NTSK_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_wRLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, R_NTSK_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    R_NTSK_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ', '.join(f"{key.replace('_', '\\_')}: {str(value).replace('_', '\\_')}" for key, value in best_NTSK_wRLS_params.items()) + ' \\\\'
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", param, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, R_NTSK_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(R_NTSK_)

# -----------------------------------------------------------------------------
# RF-NTSK
# -----------------------------------------------------------------------------


Model_Name = "RF-NTSK"

# Array to store predictions and errors
y_pred = np.array([])
RMSE_l = np.array([])
NRMSE_l = np.array([])
NDEI_l = np.array([])
MAE_l = np.array([])
MAPE_l = np.array([])
CPPM_l = np.array([])

start = 0
for train, test in expanding_window.split(X):
    
    y_test_part = y_test_exp[start:start+test.shape[0]]

    # Test the model
    y_pred_part = (RMSE_R_NTSK/(RMSE_RF + RMSE_R_NTSK)) * y_pred_RF[start:start+test.shape[0]] + (RMSE_RF/(RMSE_RF + RMSE_R_NTSK)) * y_pred_R_NTSK[start:start+test.shape[0]]
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test_part.max() - y_test_part.min())
    NRMSE_l = np.append(NRMSE_l, NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asfarray(y_test_part.flatten()))
    NDEI_l = np.append(NDEI_l, NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test_part, y_pred_part)
    MAE_l = np.append(MAE_l, MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test_part, y_pred_part)
    MAPE_l = np.append(MAPE_l, MAPE)
    
    # Count number of times the model predict a correct increase or decrease
    # Actual variation
    next_y = y_test_part[1:]
    current_y = y_test_part[:-1]
    actual_variation = (next_y - current_y) > 0.

    # Predicted variation
    next_y_pred = y_pred_part[1:]
    current_y_pred = y_pred_part[:-1]
    pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()

    # Right?
    correct = actual_variation == pred_variation

    # Correct Percentual Predictions of Movement
    CPPM = (sum(correct).item()/correct.shape[0])*100
    CPPM_l = np.append(CPPM_l, CPPM)
    
    # Compute the number of final rules
    Rules = "-"
    
    # Number of rules
    if Rules != "-":
        Rules_l = np.append(Rules_l, Rules)
    
    start += test.shape[0]

# Save predictions to dataframe
y_pred = (RMSE_R_NTSK/(RMSE_RF + RMSE_R_NTSK)) * y_pred_RF + (RMSE_RF/(RMSE_RF + RMSE_R_NTSK)) * y_pred_R_NTSK
predictions[f'{Model_Name}'] = y_pred

# Compute the mean and standard deviation of the errors
NRMSE_mean = np.mean(NRMSE_l,axis=0)
NRMSE_std = np.std(NRMSE_l,axis=0)
NDEI_mean = np.mean(NDEI_l,axis=0)
NDEI_std = np.std(NDEI_l,axis=0)
MAPE_mean = np.mean(MAPE_l,axis=0)
MAPE_std = np.std(MAPE_l,axis=0)
CPPM_mean = np.mean(CPPM_l,axis=0)
CPPM_std = np.std(CPPM_l,axis=0)

# Store outputs and error
y_pred_RF_NTSK = y_pred
RMSE_RF_NTSK = RMSE


if Rules != "-":
    Rules_mean = round(np.mean(Rules_l,axis=0))
    Rules_std = round(np.std(Rules_l,axis=0))
   
    RF_NTSK_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    param = ""
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', f'{int(Rules_mean)} ({int(Rules_std)})', "", NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RF_NTSK_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    RF_NTSK_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    param = ""
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean:.2f} ({NRMSE_std:.2f})', f'{NDEI_mean:.2f} ({NDEI_std:.2f})', f'{MAPE_mean:.2f} ({MAPE_std:.2f})', f'{CPPM_mean:.2f} ({CPPM_std:.2f})', "-", "", NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RF_NTSK_]], columns=columns)
    # Store results to dataframe
    if results.empty:
        results = newrow.copy()
    else:
        results = pd.concat([results, newrow], ignore_index=True)

print(RF_NTSK_)

#-----------------------------------------------------------------------------
# Save results to excel
#-----------------------------------------------------------------------------


# Save results
results.to_excel(f'Results/Results_{Serie}_horizon{horizon}.xlsx')

# Save predictions
predictions.to_excel(f'Predictions/Predictions_{Serie}_horizon{horizon}.xlsx')


#-----------------------------------------------------------------------------
# Print results
#-----------------------------------------------------------------------------


# Print the results
for i in results.index:
    rules = results.loc[i,"Rules"] if type(results.loc[i,"Rules"]) is int else results.loc[i,"Rules"][-1]
    print(f'{results.loc[i,"Model_Name"]} & {results.loc[i,"NRMSE"]} & {results.loc[i,"NDEI"]} & {results.loc[i,"MAPE"]} & {results.loc[i,"CPPM"]} & {results.loc[i,"Rules"]} \\\\')

for i in results.index:
    print(f'{results.loc[i,"Model_Name"]} & {results.loc[i,"Best_Params"]}')
    

#-----------------------------------------------------------------------------
# DM test
#-----------------------------------------------------------------------------


# Columns
col = predictions.columns

print("\n")
# Write columns name
for i in col:
    print(i, end=" & ")

print("\n")

print(f'Model & {col[-9]} & {col[-8]} & {col[-7]} & {col[-6]} & {col[-5]} & {col[-4]} & {col[-3]} & {col[-2]} & {col[-1]} ')
for i in range(1, len(col)-9):
    
    dm, pvalue1 = dm_test(predictions[col[0]].values, predictions[col[-9]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue2 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-9]].values, one_sided=True)
    
    dm, pvalue3 = dm_test(predictions[col[0]].values, predictions[col[-8]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue4 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-8]].values, one_sided=True)
    
    dm, pvalue5 = dm_test(predictions[col[0]].values, predictions[col[-7]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue6 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-7]].values, one_sided=True)
    
    dm, pvalue7 = dm_test(predictions[col[0]].values, predictions[col[-6]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue8 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-6]].values, one_sided=True)

    dm, pvalue9 = dm_test(predictions[col[0]].values, predictions[col[-5]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue10 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-5]].values, one_sided=True)
    
    dm, pvalue11 = dm_test(predictions[col[0]].values, predictions[col[-4]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue12 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-4]].values, one_sided=True)
    
    dm, pvalue13 = dm_test(predictions[col[0]].values, predictions[col[-3]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue14 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-3]].values, one_sided=True)

    dm, pvalue15 = dm_test(predictions[col[0]].values, predictions[col[-2]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue16 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-2]].values, one_sided=True)
    
    dm, pvalue17 = dm_test(predictions[col[0]].values, predictions[col[-1]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue18 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-1]].values, one_sided=True)

    print(f'{col[i]} & {pvalue1:.2f} & {pvalue3:.2f} & {pvalue5:.2f} & {pvalue7:.2f} & {pvalue9:.2f} & {pvalue11:.2f} & {pvalue13:.2f} & {pvalue15:.2f} & {pvalue17:.2f} \\\\')
