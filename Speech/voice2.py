import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as sw

#зададим частоты
Fs=16000
Fr=440
t=np.linspace(0,10,10*Fs)
S=np.sin(2*np.pi*Fr*t)
S=np.int16(S*30000)
sw.write("la.wav",Fs,S)