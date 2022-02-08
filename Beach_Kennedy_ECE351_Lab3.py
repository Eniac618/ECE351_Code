################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 3                                                        #
# 2/8/2022                                                     #
#                                                              #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

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
    
def f1(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = u(t[i]-2) - u(t[i]-9)
    return y

def f2(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = np.exp(-t[i])*u(t[i])
    return y

def f3(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = r(t[i]-2)*(u(t[i]-2) - u(t[i]-3)) + r(4-t[i])*(u(t[i]-3) - u(t[i]-4))
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
            

steps = .01
t = np.arange(0, 20 + steps, steps)
time = len(t);
tEx = np.arange(0, (2*t[time-1]) + steps, steps)

y1 = f1(t)
y2 = f2(t)
y3 = f3(t)

a = my_conv(y1, y2)*steps
b = my_conv(y2, y3)*steps
c = my_conv(y1, y3)*steps

conv1 = sig.convolve(y1, y2)*steps
conv2 = sig.convolve(y2, y3)*steps
conv3 = sig.convolve(y1, y3)*steps

#Plots of the three functions
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(t, y1)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Part 1 - f1(t)')

plt.subplot(1, 3, 2)
plt.plot(t, y2)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Part 1 - f2(t)')

plt.subplot(1, 3, 3)
plt.plot(t, y3)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Part 1 - f3(t)')
plt.show()

#Plots for user-defined convolution
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(tEx, a)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Convolution of f1 and f2')

plt.subplot(1, 3, 2)
plt.plot(tEx, b)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Convolution of f2 and f3')

plt.subplot(1, 3, 3)
plt.plot(tEx, c)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Convolution of f1 and f3')
plt.show()

#Plots for scipy.sigal.convolve()
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(tEx, conv1)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Scipy convolution of f1 and f2')

plt.subplot(1, 3, 2)
plt.plot(tEx, conv2)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Scipy convolution of f2 and f3')

plt.subplot(1, 3, 3)
plt.plot(tEx, conv3)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('Scipy convolution of f1 and f3')
plt.show()