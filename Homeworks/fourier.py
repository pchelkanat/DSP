import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hann
import scipy.io.wavfile as sw

fr,x = sw.read("voice.wav")
#print(fr)
#print(x)
#print(np.size(x))


x = x / (np.max(np.abs(x)))
speech=x[25000:60000]
silence=x[60000:110000]
sw.write("speech.wav", fr, speech)
sw.write("silence.wav", fr, silence)


#отображение половины ряда Фурье
fcv=np.fft.fft(np.abs(speech))
fcv=fcv[:np.int32((np.size(fcv))/2)]
fcs=np.fft.fft(np.abs(silence))
fcs=fcs[:np.int32((np.size(fcs))/2)]


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