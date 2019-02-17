import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\IBM_ADMIN\Desktop\\Machine_Learning_AZ\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Polynomial_Regression\\Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#here we don't use this coz we don't have enough data to divide it into train an dtest dataset /
# so instead of this we use the whole data to train our model.
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#visualizng the result of linear regression
plt.scatter(X,y,color='r')
plt.plot(X,lin_reg.predict(X),color='b')
plt.title("Linear regression")
plt.show()


#visualize polynomial regression
plt.scatter(X,y,color="b")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="r")
plt.title("Polynomial regression")
plt.show()

#predicting the result with linear regression
lin_reg.predict(6.5)



#predicting the result with poly reg
lin_reg2.predict(poly_reg.fit_transform(6.5))


