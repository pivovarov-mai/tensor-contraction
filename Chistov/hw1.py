import numpy as np
import time
from ncon import ncon

#Создадим тензоры

x = 5
a = np.random.rand(x,x,x)
b = np.random.rand(x,x,x,x)
c = np.random.rand(x,x,x,x)
d = np.random.rand(x,x,x)
e = np.random.rand(x,x,x)

#Решение через ncon

def nconn(a, b, c, d, e):
  return ncon((a, b, c, d, e), ([-1, 1, 5], [1, 6, 2, -2], [5, 6, 4, 7], [3, 4, -3], [2, 7, 3]))


print(nconn(a,b,c,d,e))
print(nconn(a,b,c,d,e).shape)

#Решение через библиотеку Numpy

def RSH(a, b, c, d, e):
    AC = np.tensordot(a, c, axes=([2],[0]))
    ACB = np.tensordot(AC, b, axes=([1,2],[0,1]))
    ACBE = np.tensordot(ACB, e, axes=([2,3],[1,0]))
    return np.tensordot(ACBE, d, axes=([1,2],[1,2]))

print(RSH(a,b,c,d,e))
print(RSH(a,b,c,d,e).shape)

#Решение через цикл for

def forr(a,b,c,d,e,x):
    ac = np.zeros((x, x, x, x, x))
    for i in range(x):
        for j in range(x):
            for k in range(x):
                for l in range(x):
                    for m in range(x):
                        for n in range(x): 
                            ac[i, j, k, l, m] = ac[i, j, k, l, m] + a[i, j, n] * c[n, k, l, m]
    acb = np.zeros((x, x, x, x, x))
    for i in range(x):
        for j in range(x):
            for k in range(x):
                for l in range(x):
                    for m in range(x):
                        for n in range(x):
                            for p in range(x):  
                                acb[i, j, k, l, m] = acb[i, j, k, l, m] + ac[i, n, p, j, k] * b[n, p, m, l]
    acbe = np.zeros((x, x, x, x))
    for i in range(x):
        for j in range(x):
            for k in range(x):
                for l in range(x):
                    for m in range(x):
                        for n in range(x):
                            acbe[i, j, k, l] = acbe[i, j, k, l] + acb[i, j, m, k, n] * e[n, m, l]
    acbed  = np.zeros((x, x, x))
    for i in range(x):
        for j in range(x):
            for k in range(x):
                for l in range(x):
                    for m in range(x):
                        acbed[i, j, k] =  acbed[i, j, k] + acbe[i, l, j, m] + d[m, l, k]
    return acbed

print(forr(a,b,c,d,e,x))
print(forr(a,b,c,d,e,x).shape)

#Замер времени

start = time.time()
nconn(a,b,c,d,e)
end = time.time() - start
print(f"Время выполнения с помощью библиотеки ncon = {end}")

start = time.time()
RSH(a,b,c,d,e)
end = time.time() - start
print(f"Время выполнения с помощью библиотеки numpy = {end}")

start = time.time()
forr(a,b,c,d,e,x)
end = time.time() - start
print(f"Время выполнения через цикл for = {end}")

# Вывод.
# Операция свертки через цикл for самая долгая и объемная по коду.
# С помощью библиотеки numpy самая быстрая.
# С помощью библиотеки ncon самая простая в написании по коду и быстее выполняется, чем через цикл for.
