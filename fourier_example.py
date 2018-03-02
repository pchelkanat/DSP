import matplotlib.pyplot as plt
import numpy as np

'''
##
s=np.array([1,2,-3,0,1,4,5,-1])
fc=np.fft.fft(s)
#print(fc)
#print(np.abs(fc[1:]))

sr=np.fft.ifft(fc)
#print(np.real(sr))
'''

##
Fs=1000
F=130
t=np.linspace(0,1,Fs)
S1=np.sin(2 * np.pi * F * t)
S2=np.sin(2*np.pi*2*F*t)
S=(S1+S2)/2
fcc=np.fft.fft(S)
print(fcc)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(t, S)

plt.subplot(2, 1, 2)
plt.plot(np.abs(fcc))
plt.show()
'''

##
from scipy.signal import hann

N=1000
S = hann(N)
#w=np.fft.fft(S)

plt.figure()

#plt.subplot(2, 1, 1)
plt.plot(S)

#plt.subplot(2,1,2)
#plt.plot(w)
plt.show()

'''
