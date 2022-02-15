################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 4                                                        #
# 2/15/2022                                                     #
#                                                              #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import math

def u(t):
    if t < 0:
        return 0
    if t >= 0:
        return 1
    
def r(t):
    if t < 0:
        return 0
    if t >= 0:
        return t
    
def h1(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = np.exp(-2*t[i])*(u(t[i])-u(t[i]-3))
    return y

def h2(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = u(t[i]-2) - u(t[i]-6)
    return y

def h3(t):
    f = 0.25
    w = f*2*np.pi
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = math.cos(w*t[i]) * u(t[i])
    return y

def my_conv(f1, f2):
    Nf1 = len(f1)           #variable with the length of f1
    Nf2 = len(f2)           #variable with the length of f2
    
    f1Ex = np.append(f1, np.zeros((1, Nf2-1)))  #creates an array that is the same size as f1 and f2
    f2Ex = np.append(f2, np.zeros((1, Nf1-1)))
    
    result = np.zeros(f1Ex.shape)   #creates a zero-filled array the same size as both functions
    
    for i in range((Nf2+Nf1-2)):    #goes through the length of f1 and f2
        result[i] = 0
        for j in range(Nf1):        #goes through the length of f1
            if (i-j+1 > 0):         #makes sure the loop doesn't go past 0 entries
                try: 
                    result[i] = result[i] + f1Ex[j]*f2Ex[i-j+1]     #combines the previous results with the product of the new entries
                except:
                    print(i,j)
    return result

def u_array(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = u(t[i])
    return y

def c1(t):
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        y[i] = 0.5*(((1-np.exp(-2*t[i]))*u(t[i])) - ((1-np.exp(-2*(t[i]-3)))*u(t[i]-3)))
    return y

def c2(t):
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        y[i] = r(t[i]-2)*u(t[i]-2) - r(t[i]-6)*u(t[i]-6)
    return y

def c3(t):
    f = 0.25
    w = 2*f*np.pi
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        y[i] = (1/w)*math.sin(w*t[i])*u(t[i])
    return y
            

steps = .01
t = np.arange(-10, 10 + steps, steps)
time = len(t);
tEx = np.arange(-20, (2*t[time-1]) + steps, steps)

y1 = h1(t)
y2 = h2(t)
y3 = h3(t)

step = u_array(t)

a = my_conv(y1, step)*steps
b = my_conv(y2, step)*steps
c = my_conv(y3, step)*steps

hand1 = c1(tEx)
hand2 = c2(tEx)
hand3 = c3(tEx)


#Plots of the three functions
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(t, y1)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('h1(t)')

plt.subplot(1, 3, 2)
plt.plot(t, y2)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('h2(t)')

plt.subplot(1, 3, 3)
plt.plot(t, y3)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('h3(t)')
plt.show()

#Plots for user-defined convolution
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(tEx, a)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Convolution of h1 and u(t)')

plt.subplot(1, 3, 2)
plt.plot(tEx, b)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Convolution of h2 and u(t)')

plt.subplot(1, 3, 3)
plt.plot(tEx, c)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Convolution of h3 and u(t)')
plt.show()


#hand convolutions
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(tEx, hand1)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Hand Calculation of y1')

plt.subplot(1, 3, 2)
plt.plot(tEx, hand2)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Hand Calculation of y2')

plt.subplot(1, 3, 3)
plt.plot(tEx, hand3)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Hand Calculation of y3')
plt.show()