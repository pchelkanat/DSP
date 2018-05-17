import numpy as np
import scipy.signal as sgn
import pylab as plb

order=13;
freq=[ 0.3, 0.75]


b, a=sgn.iirfilter(order, freq, btype='bandpass')

w, h = sgn.freqz(b,a)
 
plb.plot(w/np.pi,abs(h))