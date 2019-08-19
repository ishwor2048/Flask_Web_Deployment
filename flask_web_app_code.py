# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:27:29 2019

@author: Ishwor Bhusal
"""

# Flask Web API development
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Reading the CSV format dataset using pandas library
dataset = pd.read_csv('hiring.csv')

# Filling out the missing values in experience column with 0
dataset['experience'].fillna(0, inplace=True)

# filling out the missing values in test_score column with the mean values
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# Building the independent variables(training set) from the dataset
X = dataset.iloc[:, :3]

# Converting word to integer values
def convert_to_int(word):
    '''we are defining the words with numbers which can help us to work more efficiently'''
    
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 
                 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 
                 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

# Applying converted words-ints to the file using lambda function
X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

# Defining the predictive variable (y)
y = dataset.iloc[:, -1]

# There is requirement to split the data into training and test set, but since this dataset
# is very small, we are not doing that.

# Now time to apply model, we will start with linear regression model
from sklearn.linear_model import LinearRegression
# Fitting the regressor model
regressor = LinearRegression()

# Fitting the training data to the model
regressor.fit(X, y)

# Saving the model to disk with the pickle
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading the model to compare with the results
model = pickle.load(open('model.pkl', 'rb'))



