
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv("C:\\Users\\IBM_ADMIN\\Desktop\\Machine_Learning_AZ\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\data.csv")

#CHANGING THE FEATURES IN MATRIX
X=dataset.iloc[:,:-1].values  #independent variables matrix
y=dataset.iloc[:,-1].values  #dependent variable vector

#dealing with  misssing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#encoing catagorical data

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
Labelencoder_X=LabelEncoder()
X[:,0]=Labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
Labelencoder_y=LabelEncoder()
y=Labelencoder_y.fit_transform(y)


#splitting the dataset into test and train sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)



