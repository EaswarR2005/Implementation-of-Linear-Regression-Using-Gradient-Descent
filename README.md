# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the required packages.
2. Display the output values using graphical representation tools as scatter plot and graph.
3. Predict the values using predict() function.
4. Display the predicted values and end the program.

## Program  and output:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: EASWAR R
RegisterNumber:  212223230053
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X=np.c_[np.ones(len(X1)), X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors= (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
```
![ex3_op1](https://github.com/user-attachments/assets/fc540503-7642-4d1b-babc-cf046c987ffe)

```
X=(data.iloc[1:, :-2].values)
print(X)
```
![ex3_op2](https://github.com/user-attachments/assets/b55e5525-e679-4b8f-965a-34bf56d5dd70)

```
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```
![ex3_op3](https://github.com/user-attachments/assets/391b24a8-cfee-4a98-9427-ae446edfa878)

```

```


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
