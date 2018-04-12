import  bitstring as bs
import bitarray as ba

s='some string for some array'

#Bitstring
sbytes=s.encode("utf-8")
sbits=bs.BitArray(sbytes).bin

sbytes_d=int(sbits,2).to_bytes(len(sbytes),'big') #начало с MSB
msg=sbytes_d.decode('utf-8')

print(sbytes)
print(sbits)
print(sbytes_d)
print(msg)

"""
#Просто Python
digit=14
digit_bin=bin(digit) #в бинарную последовательность
digit=int(digit_bin,2) # в обратное представление
for i in range(len(s)):
    d=ord(s[i])
    d=bin(d)
    #zfill

#Bitarray
bit=ba.bitarray() #инициализация
bit.frombytes(s) #
sbit=bit.to01()

dbit=ba.bitarray()
dbit.extend(ba.bit_out)
print(dbit.tobytes(1))
"""