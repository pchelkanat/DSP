import numpy as np
import matplotlib.pyplot as plt

#зададим частоты
F1=2
F2=10
Fs=1000

t=np.linspace(0,1,Fs)

S1=np.sin(2*np.pi*F1*t)
S2=np.sin(2*np.pi*F2*t)

plt.figure()
plt.plot(t,S1, "-r", label="F = 2Hz")
plt.plot(t,S2, "-g", label="F = 10Hz")

plt.legend()
plt.title("Sin waves")

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

