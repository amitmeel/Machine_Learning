import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("C:\\Users\\IBM_ADMIN\\Desktop\\Machine_Learning_AZ\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression\\50_Startups.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
encoder=LabelEncoder()
X[:,3]=encoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#for avoiding dummy variable trap :  normally python libraries will take care of dummy variable trap
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#FITTING MULTIPLE LINEAR REGRESSION TO TRAINING SET
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)


#building the model using backward elemanation
import statsmodels.formula.api as sm
""" we want to add a ones matrix to the start of X but below line add this line to last of X.
    X=np.append(arr=X,values=np.ones((50,1)).astype(int),axis=1)
   so for adding the ones line at the start of the X matrix follow below line """
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
""" ols=ORDINARY LEAST SQUARE"""
regrssor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regrssor_OLS.summary()

X_opt=X[:,[0,1,2,3,4]]
""" ols=ORDINARY LEAST SQUARE"""
regrssor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regrssor_OLS.summary()






