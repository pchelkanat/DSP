import numpy as np
import scipy.signal as sgn
import pylab as plb

order=13;
freq=[0, 0.3, 0.5, 1]
gain= [0, 1, 1, 0]

b=sgn.firwin2(order, freq, gain)

w, h = sgn.freqz(b)

plb.figure()
plb.plot(w/np.pi,abs(h))
plb.show()