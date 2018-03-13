import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hann
import scipy.io.wavfile as sw

fs,x = sw.read("voice.wav")
#print(fs)
#print(x)
#print(np.size(x))

#
x = x / (np.max(np.abs(x)))
speech=x[24800:60000]
silence=x[60000:109600]
#sw.write("speech.wav", fr, speech)
#sw.write("silence.wav", fr, silence)

#160=10ns. разделение на окна по 10 нс
def frames(wave):
    n=int(np.size(wave)/160)
    y=np.zeros((n,160),dtype=np.float64)
    j=0
    for i in range(n):
        y[j]=wave[160*i:160*i+160]
        j+=1
    return y


def hanning(wave):
    w=frames(wave)
    v,h=np.shape(w)
    win=hann(160)
    result = np.array([])
    #по кол-ву окон v=310-silence /220-speech, а h-ширина окна = 160
    for i in range(v):
        for j in range(h):
            w[i][j]*=win[j]
            #print(w[i,j])
            #np.append(result,w[i,j])
        #print(w[i])
        np.append(result, w[i]) #НИЧЕГО НЕ ДЕЛАЕТ
    print(result,np.shape(result))
    return result

"""
a = np.array([])
b = np.array([[2,3,4],[1,0,2]])
print(np.append(a,b[1])) 
print(np.hstack((a,b[1],b[0])))
"""

#отображение половины ряда Фурье
def half_fourier(wave):
    fc=np.fft.fft(np.abs(wave))
    fc=fc[:np.int32((np.size(fc))/2)]
    return fc

def F0(wave, fs):
    max=0
    x=half_fourier(wave)[1:]
    for i in range(np.size(x)):
        if x[i]>max:
            max=x[i]
            s=i
        else: continue
    return s*fs/np.size(x)


fcv=half_fourier(speech)
fcs=half_fourier(silence)
print(F0(speech,fs))

#fcv=half_fourier(hanning(speech))
#fcs=half_fourier(hanning(silence))

#fcs=hanning(speech)

plt.figure()
plt.subplot(2,2,1)
plt.title("Speech")
plt.plot(speech)

plt.subplot(2,2,2)
plt.title("Silence")
plt.plot(silence)

plt.subplot(2,2,3)
plt.plot(fcv)

plt.subplot(2,2,4)
plt.plot(fcs)

plt.show()
