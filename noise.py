import numpy as np
import matplotlib.pyplot as plt

N=1000
#нормальное распределение
#n=np.random.randn(N)

#равномерное распределение
n=np.random.randn(N)

#Еще иное представление
#h, bins = np.histogram(n,100)

plt.figure()
plt.subplot(2,1,1) #1 строка, 2 столбца, 1-й график строим
plt.plot(n)

plt.subplot(2,1,2)
# Другое представление
#plt.hist(n,100)
#замена hist, без столбцов
#plt.plot(bins[:-1],h)
#возврат к столбцам
#plt.bar(bins[:-1],h)
plt.show()
