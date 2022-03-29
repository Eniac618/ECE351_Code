################################################################
#                                                              #
# Kennedy Beach                                                #
# ECE 351-51                                                   #
# Lab 9                                                        #
# 3/29/2022                                                    #
#                                                              #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack

def fft(x,fs):
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
    
    return freq, X_mag, X_phi

def clean_fft(x,fs):
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
    for i in range(len(X_phi)):
        if (np.abs(X_mag[i]) < 1e-10):
            X_phi[i] = 0
    
    return freq, X_mag, X_phi

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

fs = 100
T = 1/fs
t = np.arange(0, 2, T)

x1 = np.cos(2*np.pi*t)
x1_freq, x1_mag, x1_phi = fft(x1, fs)

plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t [s]')
plt.plot(t, x1)
plt.grid()
plt.title('Task 1 - FFT of x(t)=cos(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x1_freq, x1_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x1_freq, x1_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f [Hz]')
plt.ylabel('\_X(f)')
plt.stem(x1_freq, x1_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f [Hz]')
plt.stem(x1_freq, x1_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

x2 = 5*np.sin(2*np.pi*t)
x2_freq, x2_mag, x2_phi = fft(x2, fs)

plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t [s]')
plt.plot(t, x2)
plt.grid()
plt.title('Task 2 - FFT of x(t)=5sin(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x2_freq, x2_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x2_freq, x2_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f [Hz]')
plt.ylabel('\_X(f)')
plt.stem(x2_freq, x2_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f [Hz]')
plt.stem(x2_freq, x2_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

x3 = 2*np.cos((2*np.pi*2*t)-2) + (np.sin((2*np.pi*6*t)+3))**2
x3_freq, x3_mag, x3_phi = fft(x3, fs)

plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t [s]')
plt.plot(t, x3)
plt.grid()
plt.title('Task 3 - FFT of x(t)=2cos((2*pi*2*t)-2) + sin((2*pi*6*t)+3)^2')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x3_freq, x3_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x3_freq, x3_mag)
plt.xlim(-3.0, 3.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f [Hz]')
plt.ylabel('\_X(f)')
plt.stem(x3_freq, x3_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f [Hz]')
plt.stem(x3_freq, x3_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

x1_freq, x1_mag, x1_phi = clean_fft(x1, fs)

plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t [s]')
plt.plot(t, x1)
plt.grid()
plt.title('Task 4 - Clean FFT of x(t)=cos(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x1_freq, x1_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x1_freq, x1_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f [Hz]')
plt.ylabel('\_X(f)')
plt.stem(x1_freq, x1_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f [Hz]')
plt.stem(x1_freq, x1_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

x2_freq, x2_mag, x2_phi = clean_fft(x2, fs)

plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t [s]')
plt.plot(t, x2)
plt.grid()
plt.title('Task 4 - Clean FFT of x(t)=5sin(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x2_freq, x2_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x2_freq, x2_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f [Hz]')
plt.ylabel('\_X(f)')
plt.stem(x2_freq, x2_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f [Hz]')
plt.stem(x2_freq, x2_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

x3_freq, x3_mag, x3_phi = clean_fft(x3, fs)

plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t [s]')
plt.plot(t, x3)
plt.grid()
plt.title('Task 4 - Clean FFT of x(t)=2cos((2*pi*2*t)-2) + sin((2*pi*6*t)+3)^2')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x3_freq, x3_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x3_freq, x3_mag)
plt.xlim(-4.0, 4.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f [Hz]')
plt.ylabel('\_X(f)')
plt.stem(x3_freq, x3_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f [Hz]')
plt.stem(x3_freq, x3_phi)
plt.xlim(-4.0, 4.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

t = np.arange(0, 16, T)
x15 = fourier_series(15, 8, t)
x15_freq, x15_mag, x15_phi = clean_fft(x15, fs)

plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t [s]')
plt.plot(t, x15)
plt.grid()
plt.title('Task 5 - Clean FFT of the Fourier Series from Lab 8')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x15_freq, x15_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x15_freq, x15_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f [Hz]')
plt.ylabel('\_X(f)')
plt.stem(x15_freq, x15_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f [Hz]')
plt.stem(x15_freq, x15_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()