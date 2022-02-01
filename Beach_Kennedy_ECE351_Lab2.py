# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 2                                                        #
# 2/1/2022                                                    #
#                                #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'fontsize': 14})

steps = 1e-2
t1 = np.arange(0, 10 + steps, steps)

def func1(t):
    y = np.cos(t)
    return y

y = func1(t1)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t1, y)
plt.grid()
plt.ylabel('y(t) with Good Resoluation')
plt.xlabel('t')
plt.title('Part 1 - Plot of y(t) = cos(t) with Good Resolution')

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
    
def function(t, func):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = func(t[i])
    return y


def func2(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = r(t[i]) - r(t[i]-3) + 5*u(t[i]-3) - 2*u(t[i]-6) - 2*r(t[i]-6)
    
    return y

t = np.arange(-5, 10 + steps, steps)
y = function(t, r)
z = function(t, u)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, z)
plt.grid()
plt.ylabel('u(t)')
plt.xlabel('time')
plt.title('Part 2 - Plot of step function')

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('r(t)')
plt.xlabel('time')
plt.title('Part 2 - Plot of ramp function')

f = func2(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, f)
plt.grid(True)
plt.ylabel('f(t)')
plt.xlabel('time')
plt.title('Part 2 - Plot of f(t)')
plt.show()

f = func2(-t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, f)
plt.grid(True)
plt.ylabel('f(t)')
plt.xlabel('time')
plt.title('Part 3 - Plot of time reversal of f(t)')
plt.show()

f = func2(t-4)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, f)
plt.grid(True)
plt.ylabel('f(t)')
plt.xlabel('time')
plt.title('Part 3 - Plot of time shift of f(t-4)')
plt.show()

f = func2(-t-4)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, f)
plt.grid(True)
plt.ylabel('f(t)')
plt.xlabel('time')
plt.title('Part 3 - Plot of f(-t-4)')
plt.show()

f = func2(t/2)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, f)
plt.grid(True)
plt.ylabel('f(t)')
plt.xlabel('time')
plt.title('Part 3 - Plot of f(t/2)')
plt.show()

f = func2(2*t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, f)
plt.grid(True)
plt.ylabel('f(t)')
plt.xlabel('time')
plt.title('Part 3 - Plot of f(2t)')
plt.show()

f = func2(t)
dt = np.diff(t)
df = np.diff(f,axis = 0)/dt

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t[range(len(df))],df[:,0])
plt.grid(True)
plt.ylim([-10,10])
plt.xlabel('time')
plt.ylabel('df(t)/dt')
plt.title('Derivative of f(t)')
plt.show()