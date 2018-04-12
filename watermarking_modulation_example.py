import bitstring as bs
import numpy as np
import scipy.io.wavfile as sw
import matplotlib.pyplot as plt


#WTM
s="Natalya's watermark"
print(len(s))
sbytes=s.encode("utf-8")
sbits=list(bs.BitArray(sbytes).bin)
N=len(sbits)
print(sbits, N, type(sbits))

#ORIGIN
fs,origin = sw.read("Speech/voice.wav")
win=fs/100
origin=np.int64(origin)
print(origin, np.size(origin), np.shape(origin), type(origin[0]), win*N)

or_bytes=np.empty_like(origin)
print(or_bytes,np.size(or_bytes), np.shape(or_bytes), type(or_bytes[0]))

for i in range(np.size(origin)):
    or_bytes[i]=format(origin[i],'b').zfill(16)
print(or_bytes,np.size(or_bytes), np.shape(or_bytes))

or_mass=[]
for i in range(N):# from 0 to len(sbits) by 160, 152*160
    or_mass.append(np.max(or_bytes[i:i + 160]))

print(or_mass, np.shape(or_mass), type(or_mass[0]))

#B-Spline
def B_spline(win):
    t=np.linspace(0,1,win)
    #(t, np.size(t))
    spline=[]
    for i in range(160):
        if 0<=t[i]<=1/3:
            spline.append(9/2*t[i]**2)
        elif 1/3<t[i]<=2/3:
            spline.append(-9*t[i]**2+9*t[i]-3/2)
        elif 2/3<t[i]<=1:
            spline.append(9/2*(1-t[i])**2)
    return spline

x=B_spline(160)
plt.figure()

#plt.plot(x,'k')
#plt.show()