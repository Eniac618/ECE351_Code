################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 5                                                        #
# 2/22/2022                                                    #
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

def sine_method(R,L,C,t):
    y = np.zeros((len(t),1))
    alpha = -1/(2*R*C)
    omega = 0.5*np.sqrt((1/(R*C))**2 - 4/(L*C) + (0 + 1j))
    p = alpha + omega
    g = 1/(R*C)*p
    mag_g = np.abs(g)
    g_rad = np.angle(g)

    for i in range(len(t)):
        y[i] = ((mag_g/np.abs(omega))*np.exp(alpha*t[i])*np.sin(np.abs(omega)*t[i] + g_rad))*u(t[i])
    return y

steps = 1e-5
t = np.arange(0, 1.2e-3 + steps, steps)

R = 1e3
L = 27e-3
C = 100e-9
X = 1/(R*C)
Y = 1/(L*C)

num = [X, 0]
den = [1, X, Y]


h_hand = sine_method(R, L, C, t)
tout, yout = sig.impulse((num, den), T = t)

plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.plot(t, h_hand)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('h(t)')

plt.subplot(1, 2, 2)
plt.plot(tout, yout)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('time')
plt.title('sig.impulse')

tout, yout = sig.step((num,den), T = t)

plt.figure(figsize = (10,7))
plt.plot(tout,yout)
plt.xlabel('t')
plt.grid()
plt.title('Sig.step')

