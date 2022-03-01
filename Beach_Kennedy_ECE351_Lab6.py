################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 6                                                        #
# 3/1/2022                                                    #
#                                                              #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def hand_step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y[i] = (0.5-0.5*np.exp(-4*t[i])+np.exp(-6*t[i]))*u(t[i])
    return y

def cosine_method(R, P, t):
    y = np.zeros(t.shape)
    
    for i in range(len(R)):
        mag_k = np.abs(R[i])
        angle_k = np.angle(R[i])
        alpha = np.real(P[i])
        omega = np.imag(P[i])
        
        y = y + mag_k * np.exp(alpha*t) * np.cos(omega*t + angle_k) * u(t)
    return y

steps = 1e-5
t1 = np.arange(0, 2 + steps, steps)

num1 = [1, 6, 12]
den1 = [1, 10, 24]

y1 = (0.5-0.5*np.exp(-4*t1)+np.exp(-6*t1))*u(t1)
tout, yout = sig.step((num1, den1), T = t1)


plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.plot(t1, y1)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('seconds')
plt.title('y1(t) by hand')

plt.subplot(1, 2, 2)
plt.plot(tout, yout)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('seconds')
plt.title('y1(t) using sig.step')

num2 = [1, 6, 12]
den2 = [1, 10, 24, 0]

R1, P1, K1 = sig.residue(num2,den2)

print('R1: ', R1, '\nP1: ', P1, '\nK1: ', K1)

num3 = [25250]
den3 = [1, 18, 218, 2036, 9085, 25250, 0]

R2, P2, K2 = sig.residue(num3,den3)
print('R2: ', R2, '\nP2: ', P2, '\nK2: ', K2)

t2 = np.arange(0, 4.5 + steps, steps)
    
y2 = cosine_method(R2, P2, t2)

num4 = [25250]
den4 = [1, 18, 218, 2036, 9085, 25250]
tout2, yout2 = sig.step((num4, den4), T = t2)

plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.plot(t2, y2)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('seconds')
plt.title('y2(t) using cosine method')

plt.subplot(1, 2, 2)
plt.plot(tout2, yout2)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('seconds')
plt.title('y2(t) using sig.step')


