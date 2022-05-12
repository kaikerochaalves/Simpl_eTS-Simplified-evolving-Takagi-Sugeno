# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
import math

class Simpl_eTS:
    def __init__(self, InitialOmega = 1000, r = 0.1):
        self.hyperparameters = pd.DataFrame({'InitialOmega':[InitialOmega],'r':[r]})
        self.parameters = pd.DataFrame(columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Scatter', 'TimeCreation', 'NumPoints', 'Tau', 'Lambda'])
        self.ThresholdRemoveRules = 0.01
        self.InitialScatter = 0.
        self.DataScatter = 0.
        self.InitialTheta = 0.
        self.InitialPi = 0.
        self.Beta = 0.
        self.Sigma = 0.
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
         
    def fit(self, X, y):
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        # Compute z
        z = np.concatenate((x.T, y[0].reshape(-1,1)), axis=1).T
        # Initialize the first rule
        self.parameters = self.parameters.append(self.Initialize_First_Cluster(x, y[0], z), ignore_index = True)
        # Update lambda of the first rule
        self.Update_Lambda(x)
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the scatter for all rules
            for i in self.parameters.index:
                self.Update_Rule_Scatter(z, z_prev, i, k+1)
            # Compute the data scatter
            self.Update_Data_Scatter(z_prev, z, i, k+1)
            # Find the rule with the minimum and maximum scatter
            IdxMinScatter = self.parameters['Scatter'].idxmin()
            IdxMaxScatter = self.parameters['Scatter'].idxmax()
            # Compute minimum delta
            Delta = self.Minimum_Distance(z)
            # Verifying the needing to creating a new rule
            if (self.DataScatter.item() < self.parameters.loc[IdxMinScatter, 'Scatter'] or self.DataScatter.item() > self.parameters.loc[IdxMaxScatter, 'Scatter']) and Delta < 0.5 * self.hyperparameters.loc[0, 'r']:
                # Update an existing rule
                self.Rule_Update(x, z)
            elif self.DataScatter.item() < self.parameters.loc[IdxMinScatter, 'Scatter'] or self.DataScatter.item() > self.parameters.loc[IdxMaxScatter, 'Scatter']:
                # Create a new rule
                self.parameters = self.parameters.append(self.Initialize_Cluster(x, z, k+1, i), ignore_index = True)
            elif Delta > 0.5 * self.hyperparameters.loc[0, 'r']:
                # Update num points
                self.Update_Num_Points(z)
            # Remove unecessary rules
            if self.parameters.shape[0] > 1:
                self.Remove_Rule(k+1)
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
            # Store the output in the array
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        return self.OutputTrainingPhase, self.rules
            
    def predict(self, X):
        for k in range(X.shape[0]):
            x = X[k,].reshape((1,-1)).T
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Update lambda of all rules
            self.Update_Lambda(x)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
            # Store the output in the array
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase
        
    def Initialize_First_Cluster(self, x, y, z):
        NewRow = {'Center_Z': z, 'Center_X': x, 'C': self.hyperparameters.loc[0, 'InitialOmega'] * np.eye(x.shape[0] + 1), 'Theta': np.zeros((x.shape[0] + 1, 1)), 'Scatter': self.InitialScatter, 'TimeCreation': 1., 'NumPoints': 1.}
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
        return NewRow
    
    def Initialize_Cluster(self, x, z, k, i):
        Theta = np.zeros((x.shape[0] + 1, 1))
        # Update the lambda value for all rules
        self.Update_Lambda(x)
        for row in self.parameters.index:
            Theta = Theta + self.parameters.loc[row, 'Lambda'] * self.parameters.loc[row, 'Theta']
        NewRow = {'Center_Z': z, 'Center_X': x, 'C': self.hyperparameters.loc[0, 'InitialOmega'] * np.eye(x.shape[0] + 1), 'Theta': Theta, 'Scatter': self.DataScatter.item(), 'TimeCreation': k, 'NumPoints': 1}
        return NewRow
      
    def Update_Rule_Scatter(self, z, z_prev, i, k):
        self.parameters.at[i, 'Scatter'] = (((k - 2) / (k - 1)) * self.parameters.loc[i, 'Scatter']) + sum((z - z_prev)**2)
        
    def Distance(self, p1, p2):
        distance = np.linalg.norm(p1 - p2)
        return distance
    
    def Update_Data_Scatter(self, z_prev, z, i, k):
        self.Beta = self.Beta + z_prev
        self.Gamma = self.Sigma + sum(z_prev**2)
        self.DataScatter = (1 / ((k - 1) * (z.shape[0]))) * ((k - 1) * sum(z**2) - 2 * sum(z * self.Beta) + self.Gamma)
        
    def Minimum_Distance(self, z):
        dist = []
        for row in self.parameters.index:
            dist.append(np.linalg.norm(self.parameters.loc[row, 'Center_Z'] - z))
        return min(dist)
                              
    def Rule_Update(self, x, z):
        dist = []
        idx = []
        for row in self.parameters.index:
            dist.append(np.linalg.norm(self.parameters.loc[row, 'Center_Z'] - z))
            idx.append(row)
        index = idx.index(dist.index(min(dist)))
        self.parameters.at[index, 'NumPoints'] = self.parameters.loc[index, 'NumPoints'] + 1
        self.parameters.at[index, 'Center_Z'] = z
        self.parameters.at[index, 'Center_X'] = x
        self.parameters.at[index, 'Potential'] = self.DataPotential.item()
            
    def Update_Num_Points(self, z):
        dist = []
        for row in self.parameters.index:
            dist.append(np.linalg.norm(self.parameters.loc[row, 'Center_Z'] - z))
        index = dist.index(min(dist))
        self.parameters.iloc[index, 6] = self.parameters.iloc[index, 6] + 1
        
    def Update_Lambda(self, x):
        # Computing lambda
        for row in self.parameters.index:
            self.parameters.at[row, 'Tau'] = self.mu(self.parameters.loc[row, 'Center_X'], x)
        Total_Tau = sum(self.parameters['Tau'])
        for row in self.parameters.index:
            self.parameters.at[row, 'Lambda'] = self.parameters.loc[row, 'Tau'] / Total_Tau
    
    def mu(self, Center_X, x):
        tau = 1
        for j in range(x.shape[0]):
            tau = tau * (1 + (2 * (x[j,0] - Center_X[j,0]) / self.hyperparameters.loc[0, 'r'])**2)
        return 1/tau
    
    def Remove_Rule(self, k):
        N_total = 0
        for i in self.parameters.index:
            N_total = N_total + self.parameters.loc[i, 'NumPoints']
        remove = []
        for i in self.parameters.index:
            if self.parameters.loc[i, 'NumPoints'] / N_total < self.ThresholdRemoveRules:
                remove.append(i)
        if len(remove) > 0:    
            self.parameters = self.parameters.drop(remove)
    
    def RLS(self, x, y, xe):
        self.Update_Lambda(x)
        for row in self.parameters.index:
            self.parameters.at[row, 'C'] = self.parameters.loc[row, 'C'] - ((self.parameters.loc[row, 'Lambda'] * self.parameters.loc[row, 'C'] @ xe @ xe.T @ self.parameters.loc[row, 'C'])/(1 + self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'C'] @ xe))
            self.parameters.at[row, 'Theta'] = self.parameters.loc[row, 'Theta'] + (self.parameters.loc[row, 'C'] @ xe * self.parameters.loc[row, 'Lambda'] * (y - xe.T @ self.parameters.loc[row, 'Theta']))
