import numpy as np
from ncon import ncon as nc

def Manual_contraction(A, B, C, D, E):

    AB = np.zeros((i, k, a, c, e))
    ABC = np.zeros((i, k, d, h, e))
    ABCE = np.zeros((i, k, j, d, f))
    ABCED = np.zeros((i, k, j))

    for i_ in range(i):
        for k_ in range(k):
            for a_ in range(a):
                for c_ in range(c):
                    for e_ in range(e):
                        for b_ in range(b):
                            AB[i_, k_, a_, c_, e_] += A[i_, b_, a_]*B[b_, k_, c_, e_]

    for i_ in range(i):
        for k_ in range(k):
            for d_ in range(d):
                for h_ in range(h):
                    for e_ in range(e):
                        for a_ in range(a):
                            for c_ in range(c):
                                ABC[i_, k_, d_, h_, e_] += AB[i_, k_, a_, c_, e_]*C[a_, c_, d_, h_]


    for i_ in range(i):
        for k_ in range(k):
            for j_ in range(j):
                for d_ in range(d):
                    for f_ in range(f):
                        for h_ in range(h):
                            for e_ in range(e):
                                ABCE[i_, k_, j_, d_, f_] += ABC[i_, k_, d_, h_, e_]*E[f_, h_, e_]

    for i_ in range(i):
        for k_ in range(k):
            for j_ in range(j):
                for d_ in range(d):
                    for f_ in range(f):
                        ABCED[i_, k_, j_] += ABCE[i_, k_, j_, d_, f_]*D[j_, d_, f_]

    return ABCED


def np_contraction(A, B, C, D, E):

    AB = np.tensordot(A, B, axes=[[1],[0]])
    # print(AB.shape, C.shape)
    ABC = np.tensordot(AB, C, axes=[[1, 3], [0, 1]])
    # print(ABC.shape, E.shape)
    ABCE = np.tensordot(ABC, E, axes=[[2, -1], [-1, -2]])
    # print(ABCE.shape, D.shape)
    ABCED = np.tensordot(ABCE, D, axes=[[-2, -1], [-2, -1]])

    return ABCED



if __name__=="__main__":

    i= 1
    j= 2
    k= 3

    a= 4
    b= 5
    c= 6
    d= 7 
    e= 8
    f= 9
    h= 10

    A = np.random.uniform(-1, 1, (i, b, a))
    B = np.random.uniform(-1, 1, (b, k, c, e))
    C = np.random.uniform(-1, 1, (a, c, d, h))
    D = np.random.uniform(-1, 1, (j, d, f))
    E = np.random.uniform(-1, 1, (f, h, e))

    result_manual = Manual_contraction(A, B, C, D, E)

    print("Manual contraction result:\n", result_manual)

    result_np = np_contraction(A, B, C, D, E)

    print("\nNumpy contraction result:\n", result_np)

    result_nc = nc(
        [A, B, C, D, E],
        [
            # i, b, a
            [-1, 1, 2],
            # b, k, c, e
            [1, -3, 3, 4],
            # a, c, d, h
            [2, 3, 5, 6],
            # j, d, f
            [-2, 5, 7],
            # f, h, e
            [7, 6, 4]
        ]
    )

    print("\nNcon contraction result:\n", result_nc)

