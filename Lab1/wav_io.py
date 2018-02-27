import scipy.io.wavfile as sw
import numpy as np


name = 'voice.wav'

f=open(name,'rb')
[fr,dti] = sw.read(f)
f.close()

outN = dti[5:300]

print('length', len(dti))
print('sampling frequency', fr)
print('signal example', np.float32(dti[5:300]))
print('data tyoe', dti.dtype)
print('max value', np.max(dti))

s2 = dti

out_name = 'voice_our.wav'

sw.write(out_name, fr, s2)