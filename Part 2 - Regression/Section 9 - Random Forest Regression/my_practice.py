
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
dataset=pd.read_csv("C:\\Users\\IBM_ADMIN\\Desktop\\Machine_Learning_AZ\\Part 2 - Regression\\Section 9 - Random Forest Regression\\Random_Forest_Regression\\Position_Salaries.csv")
#creating input and target variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#spliting the dataset into training and test sets
"""from sklearn.model_selection import train_test_split
X_train.X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)"""

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
#coz in new python model its support fro multi dimentional array only and here we need 1D array. so wee need to reshape it
y_train=np.ravel(sc_y.fit_transform(y_train.reshape(-1, 1)))
y_test=np.ravel(sc_y.fit_transform(y_test.reshape(-1,1)))"""

#fit the random forcast to dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X,y)

#predicting the target variable
y_predict=regressor.predict(6.5)
#visualize the RandomForestRegressor results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

