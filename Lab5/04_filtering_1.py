import numpy as np
from numpy.random import randn
import scipy.signal as sgn
import pylab as plb


order=130;
freq=0.05

b=sgn.firwin(order, freq, pass_zero=True)

w, h = sgn.freqz(b)
plb.figure()
plb.plot(w/np.pi,abs(h))

sig = np.cumsum(randn(800))  # Brownian noise

print("sig",sig)
sig_ff = sgn.filtfilt(b, 1, sig)
sig_lf = sgn.lfilter(b, 1, sig)

plb.figure()
plb.plot(sig, color='silver', label='Original')
plb.plot(sig_ff, color='#3465a4', label='filtfilt')
plb.plot(sig_lf, color='#cc0000', label='lfilter')
plb.legend()
plb.show()