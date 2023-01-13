import numpy as np
from ncon import ncon
import time

# инициализация тензоров
I1 = 8
A1 = 8
F1 = 8

k1 = 1
k2 = 2
k3 = 3
k4 = 4

j = 5
l1 = 6
m1 = 7

A = np.random.uniform(-1,1,(I1,j ,k1)) # тензор А (I,j,k1)
B = np.random.uniform(-1,1,(j ,F1,l1,k2)) # тензор B (j,F,l1,k2)

C = np.random.uniform(-1,1,(k1,k2,k3,k4)) # тензор C (k1,k2,k3,k4)

D = np.random.uniform(-1,1, (A1,k3,m1)) # тензор D (A1,k3,m1)
E = np.random.uniform(-1,1,(m1,k4,l1)) # тензор E (m1,k4,l1)

print(f'A.shape {A.shape }')
print(f'B.shape {B.shape }')
print(f'C.shape {C.shape }')
print(f'D.shape {D.shape }')
print(f'E.shape {B.shape }')
# A.shape (8, 5, 1)
# B.shape (5, 8, 6, 2)
# C.shape (1, 2, 3, 4)
# D.shape (8, 3, 7)
# E.shape (5, 8, 6, 2)

# проводим свертку
# ВАРИАНТ 1
# ncon ncon(L, v, order=None, forder=None, check_indices=True):
start_ncon = time.time()
result_ncon = ncon(
    (
        A,
        B,
        C,
        D,
        E
     ),
    (
        [-1, 2, 1], # тензор А (I,j,k1)
        [2, -2,5 , 3], # тензор B (j,F,l1,k2)
        [1, 3, 6, 4], # тензор C (k1,k2,k3,k4)
        [-3, 6, 7 ], # тензор D (A,k3,m1)
        [ 7, 4,5 ] # тензор E (m1,k4,l1)
     )
)
print(time.time() - start_ncon)
print(f'result_ncon.shape = {result_ncon.shape}')
print(result_ncon)


# ВАРИАНТ 2
# np.tensordot np.tensordot(A, B, axes = (axes_A, axes_B))

start_tensordot  = time.time()
#####(1)#####
AC_1 = np.tensordot(
    A,C,
    axes=([2],[0])
)
print(f'AC_1.shape = {AC_1.shape}')
#####(2)#####
ACB_1 = np.tensordot(
    AC_1,B,
    axes=([1,2],[0,-1])
)
print(f'ACB_1.shape = {ACB_1.shape}')
#####(3)#####
ACBE_1 = np.tensordot(
    ACB_1,E,
    axes=([2,-1],[-2,-1])
)
print(f'ACBE_1.shape = {ACBE_1.shape}')
#####(4)#####
result_Tensordot = np.tensordot(
    ACBE_1,D,
    axes=([1,-1],[-2,-1])
)

print(time.time() - start_tensordot,'\n')
print(f'result_Tensordot.shape = {result_Tensordot.shape}')
print(f'result_Tensordot = {result_Tensordot}')

#check
print(result_ncon - result_Tensordot)



# ВАРИАНТ 3
start_FOR  = time.time()
#####(1)#####
AC_3 = np.zeros((I1,j ,k2,k3,k4)) #8 5 2 3 4
for i_1 in range(I1):
  for J_0 in range(j):
    for K_2 in range(k2):
      for K_3 in range(k3):
        for K_4 in range(k4):
          #sum
          for sumIter in range(k1):
            #print(sumIter)
            AC_3[i_1,J_0 ,K_2,K_3,K_4] = AC_3[i_1,J_0 ,K_2,K_3,K_4] +  \
            A[i_1][J_0][sumIter] * C[sumIter][K_2][K_3][K_4]
print(f"AC_3.shape = {AC_3.shape}")
#####(2)#####
ACB_3 = np.zeros((I1,k3,k4,F1,l1))
for i_1 in range(I1):
  for K_3 in range(k3):
    for K_4 in range(k4):
      for f_1 in range(F1):
        for L_1 in range(l1):
          #sum
          for sumIter1 in range(j):
            for sumIter2 in range(k2):
                ACB_3[i_1,K_3,K_4,f_1,L_1] = ACB_3[i_1,K_3,K_4,f_1,L_1] + \
                 AC_3[i_1,sumIter1,sumIter2,K_3,K_4] * B[sumIter1 ,f_1,L_1,sumIter2]

print(f"ACB_3.shape = {ACB_3.shape}")
#####(3)#####
ACBE_3 = np.zeros((I1,k3,F1,m1)) #= (8, 3, 4, 8, 6)  ()
# ACB_3
for i_1 in range(I1):
  for K_3 in range(k3):
      for f_1 in range(F1):
            for M_1 in range(m1):
              for sumIter1 in range(k4):
                  for sumIter2 in range(l1):
                            ACBE_3[i_1,K_3,f_1,M_1] = ACBE_3[i_1,K_3,f_1,M_1] + \
                            ACB_3[i_1,K_3,sumIter1,f_1,sumIter2] * E[M_1,sumIter1,sumIter2]

print(f"ACBE_3.shape = {ACBE_3.shape}")
####(4)#####
result_For  = np.zeros((I1,F1,A1))
for i_1 in range(I1):
  for f_1 in range(F1):
    for a_1 in range(A1):
      for sumIter1 in range(k3):
          for sumIter2 in range(m1):  #D (A,k3,m1)
              result_For[i_1,f_1,a_1] =  result_For[i_1,f_1,a_1] + \
              ACBE_3[i_1,sumIter1,f_1,sumIter2] + D[a_1,sumIter1,sumIter2]

print(f"result_For.shape = {result_For.shape}")
print(time.time() - start_FOR,'\n')

#print(f'result_For = {result_For}')

#check
print(result_ncon - result_For)

