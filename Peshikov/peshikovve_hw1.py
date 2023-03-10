# -*- coding: utf-8 -*-
"""PeshikovVE_HW1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M4S53sd2S1LD9Zi6c6qnutTrVKOZuvMp

Имеются 5 тензеров: А, B, C, D, E.<br>
Задача: осуществить свертку и выполнить ее З-мя способами:<br>
1) свернуть, используя цикл for<br>
2) вручную используя функции reshape<br>
3) используя пакет ncon<br><br>

Сравнить эти способы по:<br>
а) объему написания кода<br>
б) по времени исполнения<br>
в) проверить, что эти способы являются равнозначными (сравнения тензеров можно произвести с помощью пакета numpy)<br><br>
Итогом свертки будет некий тензор R валентности 3
"""

pip install ncon

import numpy as np
import time
from ncon import ncon

"""Обявляем тензоры"""

# Свободная нога
FreeLeg = 8
# Не Свободные ноги
Leg1 = 1
Leg2 = 2
Leg3 = 3
Leg4 = 4
Leg5 = 5
Leg6 = 6
Leg7 = 7

A = np.random.uniform(-1,1,(FreeLeg,Leg5 ,Leg1))
B = np.random.uniform(-1,1,(Leg5 ,FreeLeg,Leg6,Leg2))
C = np.random.uniform(-1,1,(Leg1,Leg2,Leg3,Leg4))
D = np.random.uniform(-1,1, (FreeLeg,Leg3,Leg7))
E = np.random.uniform(-1,1,(Leg7,Leg4,Leg6))

"""#1. С использованием цикла for"""

TimeStart  = time.time()
# Свертка 1
SV1 = np.zeros((FreeLeg, Leg5, Leg2, Leg3, Leg4))
for freeLeg in range(FreeLeg):
  for leg5 in range(Leg5):
    for leg2 in range(Leg2):
      for leg3 in range(Leg3):
        for leg4 in range(Leg4):
          for sum in range(Leg1):
            SV1[freeLeg, leg5, leg2, leg3, leg4] = SV1[freeLeg, leg5, leg2, leg3, leg4] + A[freeLeg][leg5][sum] * C[sum][leg2][leg3][leg4]
# Свертка 2
SV2 = np.zeros((FreeLeg, Leg3, Leg4, FreeLeg, Leg6))
for freeLeg in range(FreeLeg):
  for leg3 in range(Leg3):
    for leg4 in range(Leg4):
      for freeLeg1 in range(FreeLeg):
        for leg6 in range(Leg6):
          for sum in range(Leg5):
            for sum1 in range(Leg2):  
              SV2[freeLeg, leg3, leg4, freeLeg1, leg6] = SV2[freeLeg, leg3, leg4, freeLeg1, leg6] + SV1[freeLeg, sum, sum1, leg3, leg4] * B[sum, freeLeg1, leg6, sum1]
# Свертка 3
SV3 = np.zeros((FreeLeg,Leg3,FreeLeg,Leg7))
for freeLeg in range(FreeLeg):
  for leg3 in range(Leg3):
    for freeLeg1 in range(FreeLeg):
      for leg7 in range(Leg7):
        for sum in range(Leg4):
          for sum1 in range(Leg6):
            SV3[freeLeg, leg3, freeLeg1, leg7] = SV3[freeLeg, leg3, freeLeg1, leg7] + SV2[freeLeg, leg3, sum, freeLeg1, sum1] * E[leg7, sum, sum1]
# Свертка 4
SV4  = np.zeros((FreeLeg,FreeLeg,FreeLeg))
for freeLeg in range(FreeLeg):
  for freeLeg1 in range(FreeLeg):
    for freeLeg2 in range(FreeLeg):
      for sum in range(Leg3):
        for sum1 in range(Leg7):
          SV4[freeLeg, freeLeg1, freeLeg2] =  SV4[freeLeg, freeLeg1, freeLeg2] + SV3[freeLeg, sum, freeLeg1, sum1] + D[freeLeg2, sum, sum1]

TimeEnd = time.time()
print('Total time "For" = ' + str(TimeEnd - TimeStart))
print(f"result_For.shape = {SV4.shape}")
forResult = SV4

"""#2. С использованием функции reshape"""

TimeStart  = time.time()

# Свертка 1
SV1 = np.tensordot(A, C, axes = ([2], [0]))
# Свертка 2
SV2 = np.tensordot(SV1, B, axes = ([1, 2], [0, -1]))
# Свертка 3
SV3 = np.tensordot(SV2, E, axes = ([2, -1], [-2, -1]))
# Свертка 4
SV4 = np.tensordot(SV3, D, axes = ([1, -1], [-2, -1]))

TimeEnd = time.time()
print('Total time "reshape" = ' + str(TimeEnd - TimeStart))
print(f'result_Tensordot.shape = {SV4.shape}')
reshapeResult = SV4

"""#3. C использованием пакета ncon"""

TimeStart  = time.time()

VseSvertki = ncon(
    (A, B, C, D, E),
    ([-1, 2, 1], [2, -2, 5, 3], [1, 3, 6, 4], [-3, 6, 7], [7, 4, 5])
)

TimeEnd = time.time()
print('Total time "ncom" = ' + str(TimeEnd - TimeStart))
print(f'result_ncon.shape = {VseSvertki.shape}')
ncomResult = VseSvertki

"""# Проверка

# Сравнение

а)код: ncom < reshape < for<br>
б)время: reshape < ncom < for<br>

# в) сверка
"""

print(ncomResult-forResult)
print(ncomResult-reshapeResult)