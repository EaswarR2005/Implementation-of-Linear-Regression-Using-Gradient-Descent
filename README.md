# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Load the dataset from '50_Startups.csv'.
2. Extract features 'x' and target 'y'.
3. Convert 'x' to float type and standardize using StandardScaler.
4. Standardize the target 'y' using StandardScaler.
5. Add a bias term (column of ones) to the feature matrix 'x1'.
6. Initialize the parameter vector 'theta' to zeros.
7. For each iteration in the specified range (num_iters): a. Compute predictions: y_hat = x.dot(theta). b. Compute the error: error = y_hat - y. c. Update theta using the gradient descent formula: theta = theta - (learning_rate / m) * x.T.dot(error).
8. Scale new input data using the fitted scaler.
9. Add bias term (1) to the scaled new data.
10. Compute the predicted value using the new scaled data and theta: prediction = np.dot(new_data, theta).
11. Inverse scale the prediction to get the final output.
12. Print the predicted value.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: EASWAR R
RegisterNumber: 212223230053
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

X=(data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
New_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
New_Scaled=scaler.fit_transform(New_data)
prediction=np.dot(np.append(1,New_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```
### Output:

# Preview of datasets:
![ex3_op1](https://github.com/user-attachments/assets/fc540503-7642-4d1b-babc-cf046c987ffe)

# X Initialize : 
![ex3_op2](https://github.com/user-attachments/assets/b55e5525-e679-4b8f-965a-34bf56d5dd70)

# Y Initialize : 
![ex3_op3](https://github.com/user-attachments/assets/391b24a8-cfee-4a98-9427-ae446edfa878)

# X1 and Y1 Scaled Value :
![ex3_op4](https://github.com/user-attachments/assets/083f67fa-05ae-4301-b959-a4cd90b2d5eb)

![ex3_op4 1](https://github.com/user-attachments/assets/18820886-5246-4bc4-9a36-0ca43ff8a198)
# Predicted Value:

![ex3_op5](https://github.com/user-attachments/assets/c4e31142-315f-4685-88d8-298d98f1fe01)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
