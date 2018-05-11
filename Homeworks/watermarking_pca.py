import numpy as np
import scipy.io.wavfile as sw


fs, origin = sw.read("voice.wav")
origin = np.int32(origin)
print(np.shape(origin))
print(origin)

#origin = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
#a = np.array([origin])
#print(a, np.shape(a))
# print(np.shape(origin))
# 1171118=2*13*45043
p = 45043
N = 100000

alphaM = np.zeros((1, N), dtype=np.float64)
for i in range(np.shape(origin)[0] - N + 1):
    alphaM = np.vstack((alphaM, origin[0, i:i + N]))
alphaM = alphaM[1:]
# print(alphaM)

Matr = np.zeros((N, N))
print(Matr, np.shape(Matr))
for i in range(N):
    temp = np.array([alphaM[i]])  # строка
    tempM = np.dot(temp.T, temp)
    Matr = Matr + tempM
    print(tempM)
print(Matr, type(Matr[1, 1]))

vals,vecs= np.linalg.eig(Matr)
print("vals",vals)
print("vecs",vecs)

betha=vecs[:,-1]
print("betha", betha)