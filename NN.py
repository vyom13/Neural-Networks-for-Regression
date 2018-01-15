
# coding: utf-8

# In[368]:

import numpy as np
import math
import matplotlib.pyplot as plt
y = np.matrix([0.01,0.99])

def sigmoid(z):
    return (1/(1+np.exp(-z)))

#We first the a1 matrix which has 2 inputs along with the bias which is 1 by default.
a1 = np.matrix([1,0.05,0.1])
#Creating the weight matrix with random values from 0 to 1.
th1 = np.random.random((2,3))
#Creating the second weight matrix with random values from 0 to 1.
th2 = np.random.random((2,3))


#Creating the z2 matrix from th1 and a1.
def calculate_a2(th1,a1):
    z2 =[ [ None for i in range(1) ] for j in range(2) ]
    z2[0][0] = th1[0][0]*a1[0,0] + th1[0][1]*a1[0,1] + th1[0][2]*a1[0,2]
    z2[1][0] = th1[1][0]*a1[0,0] + th1[1][1]*a1[0,1] + th1[1][2]*a1[0,2] 

    #Creating a2 from z2 after applying the sigmoid function
    a2 =[ [ None for i in range(1) ] for j in range(3) ]
    a2[0][0] = 1
    a2[1][0] = sigmoid(z2[0][0])
    a2[2][0] = sigmoid(z2[1][0])
    return a2

def calculate_a3(th2,a2):
    #Creating the z3 matrix from th2 and a2.
    z3 =[ [ None for i in range(1) ] for j in range(2) ]
    z3[0][0] = th2[0][0]*a2[0][0] + th2[0][1]*a2[1][0] + th2[0][2]*a2[2][0]
    z3[1][0] = th2[1][0]*a2[0][0] + th2[1][1]*a2[1][0] + th2[1][2]*a2[2][0] 

    #Creating a3 from z3 after applying the sigmoid function
    a3 =[ [ None for i in range(1) ] for j in range(2) ]
    a3[0][0] = sigmoid(z3[0][0])
    a3[1][0] = sigmoid(z3[1][0])
    return a3


#Calculating the total cost 
def calculate_cost(y,a3):
    j = 0.5*(math.pow(y[0,0] - a3[0][0],2)+math.pow(y[0,1] - a3[1][0],2))
    return j
#Calculating the delta 3 value 
def calculate_d3(th1,th2,a1,a2,y):
    a2 = calculate_a2(th1,a1)
    a3 = calculate_a3(th2,a2)
    delta3 = [ [ None for i in range(1) ] for j in range(2) ]
    delta3 [0][0] = (a3[0][0] - y[0,0])*(a3[0][0]*(1 - a3[0][0]))
    delta3 [1][0] = (a3[1][0] - y[0,1])*(a3[1][0]*(1 - a3[1][0]))
    D3 = np.dot(delta3,a1)
    return D3
#Calculating the delta 2 value

def calculate_d2(th1,th2,a1,a2):
    a2 = calculate_a2(th1,a1)
    a3 = calculate_a3(th2,a2)
    delta2_1 = [ [ None for i in range(1) ] for j in range(2) ]
    delta2_1 [0][0] = (a3[0][0] - y[0,0])*(a3[0][0]*(1 - a3[0][0]))*th2[0][1]*(a2[1][0]*(1-a2[1][0]))
    delta2_1 [1][0] = (a3[1][0] - y[0,1])*(a3[1][0]*(1 - a3[1][0]))*th2[1][1]*(a2[0][0]*(1-a2[0][0]))
    delta2_2 = [ [ None for i in range(1) ] for j in range(2) ]
    delta2_2 [0][0] = (a3[0][0] - y[0,0])*(a3[0][0]*(1 - a3[0][0]))*th2[0][2]*(a2[1][0]*(1-a2[1][0]))
    delta2_2 [1][0] = (a3[1][0] - y[0,1])*(a3[1][0]*(1 - a3[1][0]))*th2[1][2]*(a2[0][0]*(1-a2[0][0]))
    delta2 = [ [ None for i in range(1) ] for j in range(2) ]
    delta2 [0][0] = delta2_1[0][0] + delta2_1[1][0]
    delta2 [1][0] = delta2_2[0][0] + delta2_2[1][0]
    D2 = np.dot(delta2,np.transpose(a2))
    return D2

cost_list = []
th1_list = []
th2_list = []
t1 = th1
t2 = th2
t1_1 = []
t1_2 = []
t1_3 = []
t1_4 = []
t1_5 = []
t1_6 = []
t2_1 = []
t2_2 = []
t2_3 = []
t2_4 = []
t2_5 = []
t2_6 = []

for i in range(1000):
    l = 0.50
    a2 = calculate_a2(t1,a1)
    a3 = calculate_a3(t2,a2)
    cost = calculate_cost(y,a3)
    cost_list.append(cost)
    d3 = calculate_d3(t1,t2,a1,a2,y)
    d2 = calculate_d2(t1,t2,a1,a2)
    t1_1.append(t1[0][0])
    t1_2.append(t1[0][1])
    t1_3.append(t1[0][2])
    t1_4.append(t1[1][0])
    t1_5.append(t1[1][1])
    t1_6.append(t1[1][2])
    t2_1.append(t2[0][0])
    t2_2.append(t2[0][1])
    t2_3.append(t2[0][2])
    t2_4.append(t2[1][0])
    t2_5.append(t2[1][1])
    t2_6.append(t2[1][2])
    th1_list.append(t1)
    th2_list.append(t2)
    t1 -= np.multiply(l,d2)
    t2 -= np.multiply(l,d3)


x= range(1,1001)
plt.plot(x,cost_list)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

plt.plot(x,t1_1)
plt.xlabel('Iterations')
plt.ylabel('Theta 1_1')
plt.title('Theta 1_1 vs Iteration')
plt.show()
plt.plot(x,t1_2)
plt.xlabel('Iterations')
plt.ylabel('Theta 1_2')
plt.title('Theta 1_2 vs Iteration')
plt.show()
plt.plot(x,t1_3)
plt.xlabel('Iterations')
plt.ylabel('Theta 1_3')
plt.title('Theta 1_3 vs Iteration')
plt.show()
plt.plot(x,t1_4)
plt.xlabel('Iterations')
plt.ylabel('Theta 1_4')
plt.title('Theta 1_4 vs Iteration')
plt.show()
plt.plot(x,t1_5)
plt.xlabel('Iterations')
plt.ylabel('Theta 1_5')
plt.title('Theta 1_5 vs Iteration')
plt.show()
plt.plot(x,t1_6)
plt.xlabel('Iterations')
plt.ylabel('Theta 1_6')
plt.title('Theta 1_6 vs Iteration')
plt.show()
plt.plot(x,t2_1)
plt.xlabel('Iterations')
plt.ylabel('Theta 2_1')
plt.title('Theta 2_1 vs Iteration')
plt.show()
plt.plot(x,t2_2)
plt.xlabel('Iterations')
plt.ylabel('Theta 2_2')
plt.title('Theta 2_2 vs Iteration')
plt.show()
plt.plot(x,t2_3)
plt.xlabel('Iterations')
plt.ylabel('Theta 2_3')
plt.title('Theta 2_3 vs Iteration')
plt.show()
plt.plot(x,t2_4)
plt.xlabel('Iterations')
plt.ylabel('Theta 2_4')
plt.title('Theta 2_4 vs Iteration')
plt.show()
plt.plot(x,t2_5)
plt.xlabel('Iterations')
plt.ylabel('Theta 2_5')
plt.title('Theta 2_5 vs Iteration')
plt.show()
plt.plot(x,t2_6)
plt.xlabel('Iterations')
plt.ylabel('Theta 2_6')
plt.title('Theta 2_6 vs Iteration')
plt.show()

