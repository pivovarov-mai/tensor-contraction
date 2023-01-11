import numpy as np
from ncon import ncon as nc


'''
    Tensors initialization
'''

i0 = 1
i1 = 2
i2 = 3
i3 = 4

j = 5
m = 6
h = 7

a = 8
b = 8
c = 8


A = np.random.uniform(-1, 1, (a, j, i0))
B = np.random.uniform(-1, 1, (j, c, m, i1))
C = np.random.uniform(-1, 1, (i0, i1, i2, i3))
D = np.random.uniform(-1, 1, (b, i2, h))
E = np.random.uniform(-1, 1, (h, i3, m))

print(f'Tensor A shape: {A.shape }')
print(f'Tensor B shape: {B.shape }')
print(f'Tensor C shape: {C.shape }')
print(f'Tensor D shape: {D.shape }')
print(f'Tensor E shape: {E.shape }')


"""
    NCON implementation
    Выводы: Самый простой способ реализации. Готовое решение из "коробки". Минимум кода, производительность высокая.
"""

def contract_network_ncon(A, B, C, D, E):
    return nc(
        (A, B, C, D, E), 
        ( 
            [-1,  2,  1    ], # [a,  j,  i0    ]
            [ 2, -2,  5,  3], # [j,  c,  m,  i1]
            [ 1,  3,  6,  4], # [i0, i1, i2, i3]
            [-3,  6,  7    ], # [b,  i2, h     ]
            [ 7,  4,  5    ]  # [h,  i3, m     ]
         )
    ) 

result_ncon = contract_network_ncon(A, B, C, D, E)

print('Average time for 1000 iterations: 242 µs. Speed tests placed in Lab_1_BiroIO.ipynb file.')
print(f'result_ncon.shape = {result_ncon.shape}')
print(result_ncon)


'''
    Numpy implementation
    Выводы: Данный способ сложнее с точки зрения кода и его поддержки, однако этот способ реализации выигрывает по времени исполнения.
'''

def contract_network_numpy(A, B, C, D, E, with_debug_info=False):
    
    AC_1 = np.tensordot(A, C, axes=([2], [0]))
    ACB_1 = np.tensordot(AC_1,B, axes=([1, 2], [0, -1]))
    ACBE_1 = np.tensordot(ACB_1, E, axes=([2, -1], [-2, -1]))
    result = np.tensordot(ACBE_1, D, axes=([1, -1], [-2, -1]))
    
    if not with_debug_info:
        return result
    
    print("Tensor AC_1 shape:", AC_1.shape)
    print("Tensor ACB_1 shape:", ACB_1.shape)
    print("Tensor ACBE_1 shape:", ACBE_1.shape)
    print("Tensor result shape:", result.shape)
    
    return result

print("Numpy implementation result:")
result_np = contract_network(A, B, C, D, E, with_debug_info=True)

print('Average time for 1000 iterations: 112 µs. Speed tests placed in Lab_1_BiroIO.ipynb file.')
print("Difference between Numpy and NCON implementations:")
print(result_ncon - result_np)

"""
    Native implementation
    Выводы: Самый дорогой и неоправданный способ реализации. Работает медленно, реализация слишком сложная и минимально гибкая
"""

def contract_network_native(A, B, C, D, R):
    AC_1 = np.zeros((a, j, i1, i2, i3))
    for a_i in range(a):
        for j_i in range(j):
            for i1_i in range(i1):
                for i2_i in range(i2):
                    for i3_i in range(i3):
                        for i0_i in range(i0):
                            ac_idx = a_i, j_i, i1_i, i2_i, i3_i
                            AC_1[ac_idx] = AC_1[ac_idx] + A[a_i][j_i][i0_i] * C[i0_i][i1_i][i2_i][i3_i]

    ACB_1 = np.zeros((a, i2, i3, c, m))
    for a_i in range(a):
        for i2_i in range(i2):
            for i3_i in range(i3):
                for c_i in range(c):
                    for m_i in range(m):
                        for j_i in range(j):
                            for i1_i in range(i1):  
                                acb_idx = a_i, i2_i, i3_i, c_i, m_i
                                ACB_1[acb_idx] = ACB_1[acb_idx] + AC_1[a_i, j_i, i1_i, i2_i, i3_i] * B[j_i, c_i, m_i, i1_i]


    ACBE_1 = np.zeros((a, i2, c, h))
    for a_i in range(a):
        for i2_i in range(i2):
            for c_i in range(c):
                for h_i in range(h):
                    for i3_i in range(i3):
                        for m_i in range(m):
                            abce_idx = a_i, i2_i, c_i, h_i
                            acb_idx = a_i, i2_i, i3_i, c_i, m_i
                            ACBE_1[abce_idx] = ACBE_1[abce_idx] + ACB_1[acb_idx] * E[h_i, i3_i, m_i]
        
    result  = np.zeros((a, c, b))
    for a_i in range(a):
        for c_i in range(c):
            for b_i in range(b):
                for i2_i in range(i2):
                    for h_i in range(h):
                        result[a_i, c_i, b_i] =  result[a_i, c_i, b_i] + ACBE_1[a_i, i2_i, c_i, h_i] + D[b_i, i2_i, h_i]
    
    return result

result_native = contract_network_native(A, B, C, D, E)
print("Native result shape:", result_native.shape)
print(result_native)


