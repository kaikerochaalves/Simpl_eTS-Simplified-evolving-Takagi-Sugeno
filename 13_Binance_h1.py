# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Kaike Alves
"""

# Import libraries
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from permetrics.regression import RegressionMetric
import statistics as st
from dieboldmariano import dm_test

# Neural Network
# Model
from keras.models import Sequential, Model
# Layers
from keras.layers import Input, InputLayer, Dense, Dropout, Conv1D, GRU, LSTM, MaxPooling1D, Flatten, SimpleRNN
#from tensorflow.keras.layers import Input
from tcn import TCN
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
from NMFIS import NMFIS
from NTSK import NTSK
from GEN_NMFIS import GEN_NMFIS
from GEN_NTSK import GEN_NTSK


# Binance library
from binance.client import Client


#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------


# Function to download the data
def GetHistoricalData(symbol, interval, fromDate, toDate):
    # Create a list called klines with the data
    klines = client.get_historical_klines(symbol, interval, fromDate, toDate)
    # Create a dataframe with the data
    df = pd.DataFrame(klines, columns=['dateTime', 'open', 'high', 'low', 'close', 
                                       'volume', 'closeTime', 'quoteAssetVolume', 
                                       'numberOfTrades', 'takerBuyBaseVol', 
                                       'takerBuyQuoteVol', 'ignore'])
    # Change the hour and day to datetime
    df.dateTime = pd.to_datetime(df.dateTime, unit='ms', utc=True)
    df['date'] = df.dateTime.dt.strftime("%d/%m/%Y")
    df['time'] = df.dateTime.dt.strftime("%H:%M:%S")
    # Transform object to float
    # df = df.astype({'open':'float','high':'float','low':'float','close':'float',
    #                 'volume':'float'})
    df = df.astype({'open':'float','high':'float','low':'float','close':'float',
                    'volume':'float', 'quoteAssetVolume':'float', 
                    'numberOfTrades':'float', 'takerBuyBaseVol':'float', 
                    'takerBuyQuoteVol':'float'})
    # Erase some columns
    # df = df.drop(['dateTime', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 
    #               'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
    df = df.drop(['dateTime', 'closeTime', 'ignore'], axis=1)
    # Define the order of columns index
    column_names = ["date", "time", "open", "high", "low", "close", "volume", 'quoteAssetVolume', 
    'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol']
    df = df.reindex(columns=column_names)
    return df


# Define target column
def TargetColumn(df,horizon):
    df1 = pd.DataFrame()
    df1['NextClose'] = df.loc[horizon:,'close']
    df1.reset_index(drop=True, inplace=True)
    df = df.join(df1)
    df.drop(df.index[-horizon:], inplace=True)
    return df


# Remove the date and time and fill na with the mdedium
def FillNan(df):
    df.drop(["date", "time"], axis=1, inplace=True)
    imputer = SimpleImputer(strategy = "median")
    imputer.fit(df)
    imputer.statistics_
    X = imputer.transform(df)
    df = pd.DataFrame(X, columns=df.columns, index = df.index)
    return df


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

   
Serie = "BNBUSDT"

# List of horizons
horizon = 1

# Choose the initial date
fromDate = str(datetime.strptime('01/01/2020', '%d/%m/%Y'))
# Choose the initial date
toDate = str(datetime.strptime('01/01/2022', '%d/%m/%Y'))
Interval = Client.KLINE_INTERVAL_1DAY
# Loading the login details
api_key = "yu5feGEPwjHybVZKTGePwQ05qA7dY5MeBJlJxSVwHXPyizBdjWICrMHgzrDKsmhr"
api_secret = "vaTkNgNnYXF1pUxEcv20pIPxA7Ex0UzbMrXVF5lsVSxpOAgMWCEJYUBkSa24Zp48"
client = Client(api_key, api_secret)

# Call the function to download the data
df = GetHistoricalData(Serie, Interval, fromDate, toDate)

# Prepare the data
df = FillNan(df)

# Add the target column value
Data = TargetColumn(df, horizon)

# Separate X and y
attributes = ["open", "high", "low", "close", "volume","quoteAssetVolume", 
              "numberOfTrades", "takerBuyBaseVol", "takerBuyQuoteVol"]
target = ["NextClose"]
X = Data[attributes].values
y = Data[target].values.ravel()

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

Model_Name = "KNN"

# Define Grid Search parameters
parameters = {'n_neighbors': [2, 3, 5, 10, 20]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    KNN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_KNN_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, KNN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    KNN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_KNN_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", KNN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Regression Tree
#-----------------------------------------------------------------------------

Model_Name = "Regression Tree"

# Define Grid Search parameters
parameters = {'max_depth': [2, 4, 6, 8, 10, 20, 30, 50, 100]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    RT_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_RT_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RT_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    RT_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_RT_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", RT_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Random Forest
#-----------------------------------------------------------------------------

Model_Name = "Random Forest"

# Define Grid Search parameters
parameters = {'n_estimators': [50, 250, 500, 1000, 1500, 2000],'max_depth': [10, 20, 30, 40, 50, 70,100, None],'bootstrap': [True, False]}

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    RF_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_RF_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RF_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    RF_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_RF_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", RF_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# SVM
#-----------------------------------------------------------------------------

Model_Name = "SVM"

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    SVM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_SVM_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, SVM_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    SVM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_SVM_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", SVM_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# LS-SVM
#-----------------------------------------------------------------------------

Model_Name = "LS-SVM"

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
    LS_SVR = LSSVR(kernel='linear')
    LS_SVR.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = LS_SVR.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    LS_SVR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', "", NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, LS_SVR_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    LS_SVR_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", "", NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", LS_SVR_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Gradient Boosting
#-----------------------------------------------------------------------------

Model_Name = "GB"

# Define Grid Search parameters
parameters = {'learning_rate':[0.01, 0.05, 0.1, 0.5, 0.9], 'n_estimators': [2, 4, 8, 16, 32, 64, 100, 200]}

grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    GB = GradientBoostingRegressor(**param)
    GB.fit(X_train,y_train)
    # Make predictions
    y_pred = GB.predict(X_val)
    
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
        best_GB_params = param
        
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
    GB = GradientBoostingRegressor(**best_GB_params)
    GB.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = GB.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    GB_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_GB_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GB_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GB_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_GB_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GB_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
    
# -----------------------------------------------------------------------------
# XGBoost
# -----------------------------------------------------------------------------

Model_Name = "XGBoost"

# Define Grid Search parameters
parameters = {'n_estimators':[250, 500], 'min_child_weight':[2,5], 
              'gamma':[i/10.0 for i in range(3,6)], 'max_depth': [2, 3, 4, 6, 7], 
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    XGBoost_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_XGBoost_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, XGBoost_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    XGBoost_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_XGBoost_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", XGBoost_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Light-GBM Regressor
#-----------------------------------------------------------------------------

Model_Name = "LGBM"

# Define Grid Search parameters
parameters = {'learning_rate':[0.01, 0.05, 0.1, 0.5, 0.9],'verbosity':[-1]}

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    LGBM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_LGBM_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, LGBM_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    LGBM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_LGBM_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", LGBM_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    


#-----------------------------------------------------------------------------
# Deep Learning
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# MLP
#-----------------------------------------------------------------------------


Model_Name = "MLP"

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, activation = "relu", learning_rate=3e-3, input_shape=[4]):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_MLP_parameters = {'n_hidden':0, 'n_neurons':0, 'activation':"relu", 'learning_rate': 0.318, 'input_shape': X_train_DL.shape[1]}

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
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}_part{part}_horizon{horizon}.h5', save_best_only=True)
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    MLP_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_MLP_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, MLP_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    MLP_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_MLP_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", MLP_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# CNN
#-----------------------------------------------------------------------------


Model_Name = "CNN"

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
best_CNN_parameters = {'n_hidden':3, 'n_neurons':85, 'learning_rate': 0.002}

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
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}_part{part}_horizon{horizon}.h5', save_best_only=True)
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    CNN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_CNN_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, CNN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    CNN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_CNN_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", CNN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# RNN
#-----------------------------------------------------------------------------

Model_Name = "RNN"

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
best_RNN_parameters = {'n_hidden':3, 'n_neurons':71, 'learning_rate': 4.0e-4}

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
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}_part{part}_horizon{horizon}.h5', save_best_only=True)
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    RNN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_RNN_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, RNN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    RNN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_RNN_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", RNN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
    
#-----------------------------------------------------------------------------
# LSTM
#-----------------------------------------------------------------------------

Model_Name = "LSTM"

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
best_LSTM_parameters = {'n_neurons':75, 'n_lstm_hidden':3, 'neurons_dense': 1, 'dropout_rate': 0, 'n_dense_hidden':1, 'learning_rate': 6.1e-4}

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
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}_part{part}_horizon{horizon}.h5', save_best_only=True)
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    LSTM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_LSTM_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, LSTM_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    LSTM_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_LSTM_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", LSTM_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# GRU
#-----------------------------------------------------------------------------

Model_Name = "GRU"

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
best_GRU_parameters = {'filters':2, 'kernel_size':5, 'strides': 3, 'n_neurons': 44, 'n_gru_hidden':2, 'neurons_dense': 0, 'dropout_rate': 0, 'n_dense_hidden': 0, 'learning_rate': 0.001}

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
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}_part{part}_horizon{horizon}.h5', save_best_only=True)
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    GRU_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_GRU_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GRU_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GRU_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_GRU_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GRU_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
    
    

#-----------------------------------------------------------------------------
# TCN
#-----------------------------------------------------------------------------


Model_Name = "TCN"

# Define the function to create models for the optimization method
def build_model(kernel_size = 1, dilations = [1], n_tcn_hidden = 1, neurons = 30, dropout_rate = 0, n_dense_hidden = 1, learning_rate=3e-3):
    
    i = Input(batch_shape=X_train_DL.shape)
    if n_tcn_hidden == 1:
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=False)(i)
    elif n_tcn_hidden == 2:
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=True)(i)
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=False)(o)
    else:
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=True)(i)
        for layer in range(1,n_tcn_hidden-1):
            o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=True)(o)
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=False)(o)
    for dense_layer in range(n_dense_hidden):
        o = Dense(neurons)(o)
        o = Dropout(dropout_rate)(o)
    o = Dense(1)(o)
    
    model = Model(inputs=[i], outputs=[o])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_TCN_parameters = {'kernel_size':3, 'dilations':[1,2], 'n_tcn_hidden':5, 'neurons': 40, 'dropout_rate': 0.1, 'n_dense_hidden': 2, 'learning_rate': 3.1e-4}

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
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}_part{part}_horizon{horizon}.h5', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

    # Define the model
    model = build_model(**best_TCN_parameters)

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    TCN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_TCN_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, TCN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    TCN_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_TCN_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", TCN_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# WaveNet
#-----------------------------------------------------------------------------

Model_Name = "WaveNet"

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
best_WaveNet_parameters = {'dilation_rate':(1,2), 'repeat':2, 'learning_rate': 0.004}

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
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}_part{part}_horizon{horizon}.h5', save_best_only=True)
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    WaveNet_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_WaveNet_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, WaveNet_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    WaveNet_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_WaveNet_parameters, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", WaveNet_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# evolving Fuzzy Systems
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# eTS
#-----------------------------------------------------------------------------


Model_Name = "eTS"

# Define Grid Search parameters
parameters = {'InitialOmega': [50, 100, 250, 500, 750, 1000, 10000], 'r': [0.1, 0.3, 0.5, 0.7, 0.9, 1., 5, 10, 50]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    eTS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_eTS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, eTS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    eTS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_eTS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", eTS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Simpl_eTS
#-----------------------------------------------------------------------------


Model_Name = "Simpl_eTS"

# Define Grid Search parameters
parameters = {'InitialOmega': [50, 250, 500, 750, 1000], 'r': [0.1, 0.3, 0.5, 0.7]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    Simpl_eTS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_Simpl_eTS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, Simpl_eTS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    Simpl_eTS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_Simpl_eTS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", Simpl_eTS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# exTS
#-----------------------------------------------------------------------------


Model_Name = "exTS"

# Define Grid Search parameters
parameters = {'InitialOmega': [50, 250, 500, 750, 1000], 'mu_threshold': [0.1, 0.3, 0.5, 0.7]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    exTS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_exTS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, exTS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    exTS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_exTS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", exTS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# ePL
#-----------------------------------------------------------------------------


Model_Name = "ePL"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01, 0.1, 0.5, 0.9], 'beta': [0.001, 0.005, 0.01, 0.1, 0.2], 'lambda1': [0.001, 0.01, 0.1], 's': [100, 10000], 'r': [0.1, 0.25, 0.5, 0.75]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    ePL_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_ePL_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, ePL_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    ePL_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_ePL_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", ePL_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# eMG
#-----------------------------------------------------------------------------


Model_Name = "eMG"

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    eMG_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_eMG_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, eMG_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    eMG_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_eMG_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", eMG_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# ePL+
#-----------------------------------------------------------------------------


Model_Name = "ePL_plus"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01, 0.1], 'beta': [0.01, 0.1, 0.25], 'lambda1': [0.25, 0.5, 0.75], 'omega': [100, 10000], 'sigma': [0.1, 0.25, 0.5], 'e_Utility': [0.03, 0.05], 'pi': [0.3, 0.5]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    ePL_plus_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_ePL_plus_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, ePL_plus_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    ePL_plus_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_ePL_plus_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", ePL_plus_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# ePL-KRLS-DISCO
#-----------------------------------------------------------------------------


Model_Name = "ePL_KRLS_DISCO"

# Define Grid Search parameters
parameters = {'alpha': [0.05, 0.1], 'beta': [0.01, 0.1, 0.25], 'lambda1': [0.0000001, 0.001], 'sigma': [0.5, 1, 10, 50], 'e_utility': [0.03, 0.05]}
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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    ePL_KRLS_DISCO_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_ePL_KRLS_DISCO_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, ePL_KRLS_DISCO_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    ePL_KRLS_DISCO_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_ePL_KRLS_DISCO_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", ePL_KRLS_DISCO_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# Proposed Models
#-----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# NMFIS
# -----------------------------------------------------------------------------


Model_Name = "NMFIS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20)}

grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = NMFIS(**param)
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
        best_NMFIS_params = param

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
    model = NMFIS(**best_NMFIS_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
   
    NMFIS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_NMFIS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, NMFIS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    NMFIS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_ePL_KRLS_DISCO_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", NMFIS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# NTSK-RLS
# -----------------------------------------------------------------------------

Model_Name = "NTSK-RLS"

# Set hyperparameters range
parameters = {'n_clusters':[1], 'lambda1':[0.95,0.96,0.97,0.98,0.99], 'RLS_option':[1]}

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_NTSK_RLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, NTSK_RLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    NTSK_RLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_NTSK_RLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", NTSK_RLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------

Model_Name = "NTSK-wRLS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20), 'RLS_option':[2]}

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_NTSK_wRLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, NTSK_wRLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    NTSK_wRLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_NTSK_wRLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", NTSK_wRLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# GEN_NMFIS
# -----------------------------------------------------------------------------


Model_Name = "GEN_NMFIS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20)}

grid = ParameterGrid(parameters)

higher_cppm = -np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NMFIS(**param)
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
        best_GEN_NMFIS_params = param

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
    model = GEN_NMFIS(**best_GEN_NMFIS_params)
    model.fit(X_train_part,y_train_part)
    # Make predictions
    y_pred_part = model.predict(X_test_part)
    y_pred = np.append(y_pred, y_pred_part.flatten())
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test_part, y_pred_part))
    RMSE_l = np.append(RMSE_l, RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
    Rules = model.model.parameters.shape[0]
    
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
   
    GEN_NMFIS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & {Rules_mean} $\pm$ {Rules_std}'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_GEN_NMFIS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GEN_NMFIS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GEN_NMFIS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_ePL_KRLS_DISCO_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GEN_NMFIS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# GEN-NTSK-RLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-RLS"

# Set hyperparameters range
parameters = {'n_clusters':[1], 'lambda1':[0.95,0.96,0.97,0.98,0.99], 'RLS_option':[1]}

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
    Rules = model.model.parameters.shape[0]
    
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
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_GEN_NTSK_RLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GEN_NTSK_RLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GEN_NTSK_RLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_GEN_NTSK_RLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GEN_NTSK_RLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# GEN-NTSK-wRLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-wRLS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20), 'RLS_option':[2]}

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
    NRMSE = RegressionMetric(y_test_part, y_pred_part).normalized_root_mean_square_error()
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
    Rules = model.model.parameters.shape[0]
    
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
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', f'{Rules_mean} ({Rules_std})', best_GEN_NTSK_wRLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, Rules_l, GEN_NTSK_wRLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)
    
else:
   
    GEN_NTSK_wRLS_ = f'{Model_Name} & {NRMSE_mean:.2f} $\pm$ {NRMSE_std:.2f} & {NDEI_mean:.2f} $\pm$ {NDEI_std:.2f} & {MAPE_mean:.2f} $\pm$ {MAPE_std*100:.2f} & {CPPM_mean:.2f} $\pm$ {CPPM_std:.2f} & "-"'

    # Store results to dataframe
    newrow = pd.DataFrame([[Model_Name, f'{NRMSE_mean} ({NRMSE_std})', f'{NDEI_mean} ({NDEI_std})', f'{MAPE_mean} ({MAPE_std})', f'{CPPM_mean} ({CPPM_std})', "-", best_GEN_NTSK_wRLS_params, NRMSE_l, NDEI_l, MAPE_l, CPPM_l, "-", GEN_NTSK_wRLS_]], columns=columns)
    results = pd.concat([results, newrow], ignore_index=True)

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
    print(f'\n{i}:')
    print(f'{results.loc[i,"Summary_Results"]}')
    print(f'{results.loc[i,"Best_Params"]}')
    

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

print(f'Model & {col[-6]} & {col[-5]} & {col[-4]} & {col[-3]} & {col[-2]} & {col[-1]}')
for i in range(1, len(col)-6):
    print("\n")
    
    dm, pvalue1 = dm_test(predictions[col[0]].values, predictions[col[-6]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue2 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-6]].values, one_sided=True)
    
    dm, pvalue3 = dm_test(predictions[col[0]].values, predictions[col[-5]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue4 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-5]].values, one_sided=True)

    dm, pvalue5 = dm_test(predictions[col[0]].values, predictions[col[-4]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue6 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-4]].values, one_sided=True)
    
    dm, pvalue7 = dm_test(predictions[col[0]].values, predictions[col[-3]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue8 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-3]].values, one_sided=True)
    
    dm, pvalue9 = dm_test(predictions[col[0]].values, predictions[col[-2]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue10 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-2]].values, one_sided=True)

    dm, pvalue11 = dm_test(predictions[col[0]].values, predictions[col[-1]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue12 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-1]].values, one_sided=True)

    print(f'{col[i]} & {pvalue1:.2f} & {pvalue3:.2f} & {pvalue5:.2f} & {pvalue7:.2f} & {pvalue9:.2f} & {pvalue11:.2f}')
