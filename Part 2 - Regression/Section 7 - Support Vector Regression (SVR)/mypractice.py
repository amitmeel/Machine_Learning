#SUPPORT VECTOR REGRESSION (SVR)

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read the dataset
dataset=pd.read_csv("C:\\Users\\IBM_ADMIN\\Desktop\\Machine_Learning_AZ\\Part 2 - Regression\\Section 7 - Support Vector Regression (SVR)\\SVR\\Position_Salaries.csv")

""" #creating inpur and target variable
#X is the input varibale , y is the target variale, here we need to predict the salary of any employee
#based on their position and level. so basically we don't need to take both position and level as input coz position already based on the level. so we just take level as input.
#basicaly this X=dataset.iloc[:,1].values will create a vector and we need matrix so change it to """

X=dataset.iloc[:,1:-1].values  #matrix(10,1)
y=dataset.iloc[:,2].values  #vector (10,)

"""aslo we dont need to create train and test samples coz there are already very less sample so we take all the samples for the training purpose."""
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))  


#fitting SVR t dataset
from sklearn.svm import SVR
regressor=SVR(kernel="rbf")  #creating the object of SVR class
regressor.fit(X,y)  #fitting the data

#predicting the new result
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()









