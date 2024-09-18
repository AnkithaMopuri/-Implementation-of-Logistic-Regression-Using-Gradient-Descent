# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize `theta` to zeros.
2. Compute `z = X * theta` and apply sigmoid to get predictions.
3. Calculate gradient and update `theta` using gradient descent.
4. Repeat for specified iterations to optimize `theta`.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MOPURI ANKITHA
RegisterNumber: 212223040117
*/
```
```
import numpy as np

def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(iterations):
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))  # Sigmoid inside loop
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
    return theta

# Example usage
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([0, 0, 1, 1])
theta = gradient_descent(X, y, 0.1, 1000)
print(theta)

```

## Output:
![image](https://github.com/user-attachments/assets/9a6e7fc5-582b-475a-b9b0-d2fa1429d91c)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

