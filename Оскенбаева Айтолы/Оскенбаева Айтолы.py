from ncon import ncon
import numpy as np
import time

# init tensors

x = 10
a = np.random.rand(x,x,x)
b = np.random.rand(x,x,x,x)
c = np.random.rand(x,x,x,x)
d = np.random.rand(x,x,x)
e = np.random.rand(x,x,x)


def tensor_for(a,b,c,d,e,x):
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

def tensor_reshape(a,b,c,d,e):
    ac = np.tensordot(a,c, axes = ([2],[0]))
    acb = np.tensordot(ac,b, axes = ([1,2],[0,1]))
    acbe = np.tensordot(acb,e, axes = ([2,3],[1,0]))
    acbed = np.tensordot(acbe,d, axes = ([1,2],[1,2]))
    return acbed

def tensor_ncon(a,b,c,d,e):
    return ncon((a, b, c, d, e), ([-1, 1, 5], [1, 6, 2, -2], [5, 6, 4, 7], [3, 4, -3], [2, 7, 3]))

## for 

start = time.time()
print(tensor_for(a,b,c,d,e,x))
for_end = time.time() - start
print(f'Time of completion: {for_end}')

## reshape

start = time.time()
print(tensor_reshape(a,b,c,d,e))
reshape_end = time.time() - start
print(f'Time of completion: {reshape_end}')

## ncon

start = time.time()
print(tensor_ncon(a,b,c,d,e))
ncon_end = time.time() - start
print(f'Time of completion: {ncon_end}')

## compare

print(f'For time: {for_end} \nreshape_time: {reshape_end} \nncon time: {ncon_end}')