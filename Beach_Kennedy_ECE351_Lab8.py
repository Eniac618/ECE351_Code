################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 8                                                        #
# 3/22/2022                                                    #
#                                                              #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt

def a_k(k):
    return 0;

def b_k(k):
    return (2/(k*np.pi))*(1 - np.cos(k*np.pi))

def fourier_series(N,T,t):
    omega = 2*np.pi/T
    y = np.zeros(t.shape)
    for i in range(1,N+1):
        y = y + (b_k(i)*np.sin(i*omega*t))
    return y

print("a0: ", a_k(0), "\na1: ", a_k(1))
print("\nb1: ", b_k(1), "\nb2: ", b_k(2), "\nb3: ", b_k(3))


steps = 1e-5
t = np.arange(0, 20 + steps, steps)
x1 = fourier_series(1,8,t)
x3 = fourier_series(3,8,t)
x15 = fourier_series(15,8,t)
x50 = fourier_series(50,8,t)
x150 = fourier_series(150,8,t)
x1500 = fourier_series(1500,8,t)

plt.figure(figsize=(20, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x1)
plt.grid(True)
plt.ylabel('N = 1')
plt.xlabel('seconds')
plt.title('x(t)')

plt.subplot(3, 1, 2)
plt.plot(t, x3)
plt.grid(True)
plt.ylabel('N = 3')
plt.xlabel('seconds')

plt.subplot(3, 1, 3)
plt.plot(t, x15)
plt.grid(True)
plt.ylabel('N = 15')
plt.xlabel('seconds')

plt.figure(figsize=(20, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x50)
plt.grid(True)
plt.ylabel('N = 50')
plt.xlabel('seconds')
plt.title('x(t)')

plt.subplot(3, 1, 2)
plt.plot(t, x150)
plt.grid(True)
plt.ylabel('N = 150')
plt.xlabel('seconds')

plt.subplot(3, 1, 3)
plt.plot(t, x1500)
plt.grid(True)
plt.ylabel('N = 1500')
plt.xlabel('seconds')