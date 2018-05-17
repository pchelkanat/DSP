# fir filter design

import numpy as np
import scipy.signal as sgn
import pylab as plb

order=3;
freq=0.3

b=sgn.firwin(order, freq, pass_zero=False)

w, h = sgn.freqz(b)

plb.figure()
plb.plot(w/np.pi,abs(h))
plb.show()