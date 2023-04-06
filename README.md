# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Dhanush.G.R.
RegisterNumber:  212221040038
*/
```
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

print("Profit Prediction Graph:")

plt.scatter(data[0],data[1])

plt.xticks(np.arange(5,30,step=5))

plt.yticks(np.arange(-5,30,step=5))

plt.xlabel("Population of City (10,000s)")

plt.ylabel("Profit ($10,000)")

plt.title("Profit Prediction")
def computeCost(X,y,theta):

  """
 
 Take in numpy array X,y,theta and generate the cost function in a linear regression model
 
 """
 
 m=len(y)
 
 h=X.dot(theta)
 
 square_err=(h-y)**2
 
 return 1/(2*m) * np.sum(square_err)

data_n=data.values

m=data_n[:,0].size

X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)

y=data_n[:,1].reshape(m,1)

theta=np.zeros((2,1))

print("Compute Cost Value:")

computeCost(X,y,theta)#call the function

def gradientDescent(X,y,theta,alpha,num_iters):

m=len(y)

J_history=[]

for i in range(num_iters):

predictions=X.dot(theta)

error=np.dot(X.transpose(),(predictions -y))

descent=alpha * 1/m * error

theta-=descent

J_history.append(computeCost(X,y,theta))

return theta,J_history

print("h(x) value:")

theta,J_history=gradientDescent(X,y,theta,0.01,1500)

print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

print("Cost function using Gradient Descent:")

plt.plot(J_history)

plt.xlabel("Iteration")

plt.ylabel("$J(\Theta)$")

plt.title("Cost function using Gradient Descent")

print("Profit Prediction:")

plt.scatter(data[0],data[1])

x_value=[x for x in range(25)]

y_value=[y*theta[1]+theta[0] for  y in x_value]

plt.plot(x_value,y_value,color="r")

plt.xticks(np.arange(5,30,step=5))

plt.yticks(np.arange(-5,30,step=5))

plt.xlabel("Population of City (10,000)")

plt.ylabel("Profit ($10,000)")

plt.title("Profit Prediction")

def predict(x,theta):

predictions=np.dot(theta.transpose(),x)

return predictions[0]

print("Profit for the Population 35,000:")

predict1=predict(np.array([1,3.5]),theta)*1000

print("For population = 35,000 we predict a profit of $"+str(round(predict1,0)))

print("Profit for the Population 70,000:")

predict2=predict(np.array([1,7]),theta)*1000

print("For population = 70,000 we predict a profit of $"+str(round(predict2,0)))

## Output:
![image](https://user-images.githubusercontent.com/128135558/230436625-3353f206-d942-4ade-986b-9a463f5a8e00.png)

![image](https://user-images.githubusercontent.com/128135558/230436758-116b07c6-2458-4456-9299-b40f40f1a5e4.png)

![image](https://user-images.githubusercontent.com/128135558/230436837-45419166-7081-4f29-805a-befe41bee8a7.png)

![image](https://user-images.githubusercontent.com/128135558/230437094-d28f12e4-b625-48db-9bed-c7d7e558c9da.png)

![image](https://user-images.githubusercontent.com/128135558/230437235-d24194b2-2ab7-44cd-8c96-562a37c68174.png)

![image](https://user-images.githubusercontent.com/128135558/230437413-17fabc52-e731-4fce-93e6-86bb0b0adaba.png)

![image](https://user-images.githubusercontent.com/128135558/230437491-c71f3506-4ceb-4709-a217-b883f68eb861.png)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
