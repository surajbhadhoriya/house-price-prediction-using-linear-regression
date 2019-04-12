#LINEAR REGRESSION MODEL   
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#Load Dataset
data1=pd.read_csv('C:/Users/SURAJ BHADHORIYA/Desktop/SINGLE-FILES/kc_house_data.csv')
# data.info()
x=np.array(data1['sqft_living'],dtype=np.float64)
y=np.array(data1['price'],dtype=np.float64)
#Visualization of data
plt.scatter(x,y,color='green')
#split dataset
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=5)
#Reshape dataset
X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
#Apply Linear Regression model
model=LinearRegression()
model.fit(X_train, y_train)
b0=model.intercept_
m=model.coef_
print(b0)
print(m)
y_pred=model.predict(X_test)
ac=model.score(X_test,y_test)
print(ac)



#========End of programe==========