import scipy.io.wavfile as sw
import scipy.signal as sgn
import numpy as np
import pylab as plb


order=13
freq=0.5

b,a=sgn.butter(order, freq, btype='high')

w, h = sgn.freqz(b,a)
 
plb.plot(w/np.pi,abs(h))

name = 'voice.wav'

f=open(name,'rb')
[fr,dti] = sw.read(f)
f.close()

dti=np.float32(dti)/32767.0

outN = sgn.filtfilt(b,a,dti)

outN=np.int16(np.round(outN*32767.0))

fwrt = open('sndNew3.wav','wb')

sw.write(fwrt, 16000, outN)
fwrt.close()