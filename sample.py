import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df1= pd.read_csv("titanic (4).csv")
df=pd.read_csv("sampledata.csv")
  

x= df[["Temperature (C)"]]
y= df[["Humidity"]]
X_train,X_test,Y_train,Y_test=train_test_split(x, y, train_size= 0.2,random_state=42)
lr= LinearRegression()
lr.fit(X_train,Y_train)
Y_pred= lr.predict(X_test)

r2=r2_score(Y_test,Y_pred)
print("r score  ",r2)
print("accuracy ",r2*100)
a= np.array([[45]])
# print(lr.predict(a))

print(df.head())



