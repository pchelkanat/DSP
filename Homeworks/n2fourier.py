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
speech=x[26000:26320]
silence=x[70000:70320]
print("Size of speech & silence: ",np.size(speech),np.size(silence))
#sw.write("speech.wav", fr, np.int16(speech))
#sw.write("silence.wav", fr, np.int16(silence))


"""
#160=10ns. разделение вольшого куска на окна по 10 нс
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
    #по кол-ву окон v, а h-ширина окна = 160
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
"""
a = np.array([])
b = np.array([[2,3,4],[1,0,2]])
print(np.append(a,b[1])) 
print(np.hstack((a,b[1],b[0])))
"""


def hanning2(wave):
    w=hann(np.size(wave))
    for i in range(np.size(wave)):
        wave[i]*=w[i]
    return wave


#отображение половины ряда Фурье
def half_fourier(wave):
    fc=np.fft.fft(np.abs(wave))
    fc=fc[:np.int32((np.size(fc))/2)]
    return fc

def F0(wave, fs):
    max=0
    #отсечь 0-й
    x=half_fourier(wave)[1:]
    for i in range(np.size(x)):
        if x[i]>max:
            max=x[i]
            s=i
        else: continue
    return s*fs/(np.size(x)+1)/2 #s, fs, np.size(x)+1


fcv=half_fourier(hanning2(speech))
fcs=half_fourier(hanning2(silence))
print("F0: ",F0(speech,fs))

plt.figure()
plt.subplot(2,2,1)
plt.title("Speech")
#plt.plot(hanning2(speech))
plt.plot(speech)

plt.subplot(2,2,2)
plt.title("Silence")
#plt.plot(hanning2(silence))
plt.plot(silence)

plt.subplot(2,2,3)
plt.plot(fcv)

plt.subplot(2,2,4)
plt.plot(fcs)

plt.show()
