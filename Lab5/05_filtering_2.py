import numpy as np
import scipy.signal as sgn
import scipy.io.wavfile as sw
import pylab as plb


order=21;
freq=0.6

b=sgn.firwin(order, freq, pass_zero=False)

name = 'voice.wav'

f=open(name,'rb')
[fr,dti] = sw.read(f)
f.close()

dti=np.float32(dti)/32767.0

outN = sgn.lfilter(b,1,dti)

outN=np.int16(np.round(outN*32767.0))

fwrt = open('sndNew2.wav','wb')

sw.write(fwrt, 16000, outN)
fwrt.close()