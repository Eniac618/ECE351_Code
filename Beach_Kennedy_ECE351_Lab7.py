################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 7                                                        #
# 3/8/2022                                                    #
#                                                              #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

numG = [1,9]
denG = sig.convolve([1,-6,-16], [1,4])

numA = [1,4]
denA = [1,4,3]

B = [1,26,168]

G_z, G_p, G_k = sig.tf2zpk(numG, denG)
print('G zeros: ', G_z, '\nG poles: ', G_p)

A_z, A_p, A_k = sig.tf2zpk(numA, denA)
print('\nA zeros: ', A_z, '\nA poles: ', A_p)

B_roots = np.roots(B)
print('\nB roots: ', B_roots)

numOpen = sig.convolve(numA, numG)
denOpen = sig.convolve(denA, denG)
print('\nOpen num: ', numOpen, '\nOpen den: ', denOpen)

open_z, open_p, open_k = sig.tf2zpk(numOpen, denOpen)
print('\nOpen loop zeros: ', open_z, '\nOpen loop poles: ', open_p)

steps = 1e-5
t = np.arange(0, 5 + steps, steps)

tout, yout = sig.step((numOpen, denOpen), T = t)

plt.figure(figsize = (10, 7))
plt.ylabel('h(t)')
plt.xlabel('t')
plt.plot(tout, yout)
plt.grid()
plt.suptitle('Step Response of the Open Loop')

numClosed = sig.convolve(numA,numG)
denClosed = sig.convolve((denG + sig.convolve(B, numG)), denA)
print('\nClosed num: ', numClosed, '\nClosed den: ', denClosed)

closed_z, closed_p, closed_k = sig.tf2zpk(numClosed, denClosed)
print('\nClosed loop zeros: ', closed_z, '\nClosed loop poles: ', closed_p)

tout2, yout2 = sig.step((numClosed, denClosed), T = t)

plt.figure(figsize = (10, 7))
plt.ylabel('h(t)')
plt.xlabel('t')
plt.plot(tout2, yout2)
plt.grid()
plt.suptitle('Step Response of the Closed Loop')
