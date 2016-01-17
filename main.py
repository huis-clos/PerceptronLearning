import pandas as pd
import perceptron
import numpy as np


def add_name2(an_array, a):
    j = 0

    for i in a:
        an_array[j].name_update(i)
        j += 1


error = 0
correct = 0
total = 0
recal = False
alpha = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']

df = pd.read_csv('letter-recognition.data', index_col=0, header=None, names=
['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'])

df.index.name = 'letter'

df = df.div(15)

train, test = np.split(df, 2)



length, width = train.shape

A = [perceptron.PerceptronA(0.2) for i in range(25)]

add_name2(A, alpha)

alpha.pop(0)

B = [perceptron.PerceptronB(0.2) for i in range(24)]

add_name2(B, alpha)

alpha.pop(0)

C = [perceptron.PerceptronC(0.2) for i in range(23)]

add_name2(C, alpha)

alpha.pop(0)

D = [perceptron.PerceptronD(0.2) for i in range(22)]

add_name2(D, alpha)

alpha.pop(0)

E = [perceptron.PerceptronE(0.2) for i in range(21)]

add_name2(E, alpha)

alpha.pop(0)

F = [perceptron.PerceptronF(0.2) for i in range(20)]

add_name2(F, alpha)

alpha.pop(0)

G = [perceptron.PerceptronG(0.2) for i in range(19)]

add_name2(G, alpha)

alpha.pop(0)

H = [perceptron.PerceptronH(0.2) for i in range(18)]

add_name2(H, alpha)

alpha.pop(0)

I = [perceptron.PerceptronI(0.2) for i in range(17)]

add_name2(I, alpha)

alpha.pop(0)

J = [perceptron.PerceptronJ(0.2) for i in range(16)]

add_name2(J, alpha)

alpha.pop(0)

K = [perceptron.PerceptronK(0.2) for i in range(15)]

add_name2(K, alpha)

alpha.pop(0)

L = [perceptron.PerceptronL(0.2) for i in range(14)]

add_name2(L, alpha)

alpha.pop(0)

M = [perceptron.PerceptronM(0.2) for i in range(13)]

add_name2(M, alpha)

alpha.pop(0)

N = [perceptron.PerceptronN(0.2) for i in range(12)]

add_name2(N, alpha)

alpha.pop(0)

O = [perceptron.PerceptronO(0.2) for i in range(11)]

add_name2(O, alpha)

alpha.pop(0)

P = [perceptron.PerceptronP(0.2) for i in range(10)]

add_name2(P, alpha)

alpha.pop(0)

Q = [perceptron.PerceptronQ(0.2) for i in range(9)]

add_name2(Q, alpha)

alpha.pop(0)

R = [perceptron.PerceptronR(0.2) for i in range(8)]

add_name2(R, alpha)

alpha.pop(0)

S = [perceptron.PerceptronS(0.2) for i in range(7)]

add_name2(S, alpha)

alpha.pop(0)

T = [perceptron.PerceptronT(0.2) for i in range(6)]

add_name2(T, alpha)

alpha.pop(0)

U = [perceptron.PerceptronU(0.2) for i in range(5)]

add_name2(U, alpha)

alpha.pop(0)

V = [perceptron.PerceptronV(0.2) for i in range(4)]

add_name2(V, alpha)

alpha.pop(0)

W = [perceptron.PerceptronW(0.2) for i in range(3)]

add_name2(W, alpha)

alpha.pop(0)

X = [perceptron.PerceptronX(0.2) for i in range(2)]

add_name2(X, alpha)

alpha.pop(0)

Y = [perceptron.PerceptronY(0.2) for i in range(1)]

add_name2(Y, alpha)

alpha.pop(0)


for i, row in enumerate(train.values):
    row = np.insert(row, 0, 1)
    for x in A:
         x.set_test_data(row, train.index[i])
         if recal:
             x.reweight()
             recal = False
         result = x.test()
         ++total
         if result == x.target:
             ++correct
         else:
            ++error
            x.reweight()
            recal = True

    break
