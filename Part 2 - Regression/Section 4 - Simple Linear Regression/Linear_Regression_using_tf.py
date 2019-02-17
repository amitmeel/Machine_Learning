# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 20:14:56 2018

@author: amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil

 # Importing the dataset
dataset = pd.read_csv(r'C:\Users\user\Desktop\Project\Machine_Learning_AZ\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv')
# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['f', 

X = dataset.iloc[:,:-1].values  #matrix (30,1)
y = dataset.iloc[:,1].values   #vector (30,)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Input functions to read from Pandas Dataframe 
def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )
  
def make_prediction_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )
  
 #Create feature columns for estimator 
def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in dataset.columns]
  return input_columns

tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time


#Linear Regression with tf.Estimator framework 
model = tf.estimator.LinearRegressor(
      feature_columns = make_feature_cols(), model_dir = OUTDIR)

model.train(input_fn = make_input_fn(df_train, num_epochs = 10))



#Evaluate on the validation data (we should defer using the test data to after we have selected a final model).
def print_rmse(model, name, df):
  metrics = model.evaluate(input_fn = make_input_fn(df, 1))
  print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
print_rmse(model, 'validation', df_valid)

# Let's use this model for prediction.
predictions = model.predict(input_fn = make_prediction_input_fn(df_valid, 1))
for i in range(5):
  print(next(predictions))