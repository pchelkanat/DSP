import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile as sw

fr,x = sw.read("voice.wav")
print(fr)
print(x)



#fr = 16000Hz
#x = amplitudes 'frames' int16 = -32756..32755
#scipy.audiolah as like as x np.float32
#x=x/(2**15-1)

fc=np.fft.fft(x[20000:60000])

plt.figure()
plt.subplot(2,1,1)
plt.plot(x[20000:60000])

plt.subplot(2,1,2)
plt.plot(fc)
#xn=x*2#громкость
#frn=np.int(fr*1.3)#частота
#sw.write("name1.wav", frn, xn)
plt.show()


