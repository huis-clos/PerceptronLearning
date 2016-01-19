# Colleen Toth
# CS545, Machine Learning
# Prof. Melanie Mitchell
# Homework 1
# Due: 19 January 2016
#
# Main perceptron testing program. Initializes perceptrons, runs
# training and then runs the test on final weights. Finally, generates
# confusion matrix based on the testing classifications. Training
# runs

import copy
import pandas as pd
import perceptron
import numpy as np
import os.path
import math as m
from os import listdir
import operator

def add_name2(an_array, a):
    j = 0

    for i in a:
        an_array[j].name_update(i)
        j += 1


def shuffle_and_write(filename, df):

    for i in range(100):
        df = df.reindex(np.random.permutation(df.index))

    df.to_csv(filename, header=None)

e_count = 0
total = correct = error = 0
recal = False
alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']
minus_Z = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y']
alpha_c = copy.deepcopy(alpha)
alpha_c.pop(0)
prev_accuracy = 0
current_accuracy = 0
letters = []
y_true = []
y = []
votes = dict.fromkeys(alpha, 0)
test_votes = dict.fromkeys(alpha, 0)
c = 0

# initialize perceptrons with eta of 0.2 for groups of A's, B's, C's
# etc. Sets initial weights as well. See perceptron.py for class
# definition and associated functions
A = [perceptron.PerceptronA(0.2) for i in range(25)]

add_name2(A, alpha_c)

alpha_c.pop(0)

letters.append(A)

B = [perceptron.PerceptronB(0.2) for i in range(24)]

add_name2(B, alpha_c)

alpha_c.pop(0)

letters.append(B)

C = [perceptron.PerceptronC(0.2) for i in range(23)]

add_name2(C, alpha_c)

alpha_c.pop(0)

letters.append(C)

D = [perceptron.PerceptronD(0.2) for i in range(22)]

add_name2(D, alpha_c)

alpha_c.pop(0)

letters.append(D)

E = [perceptron.PerceptronE(0.2) for i in range(21)]

add_name2(E, alpha_c)

alpha_c.pop(0)

letters.append(E)

F = [perceptron.PerceptronF(0.2) for i in range(20)]

add_name2(F, alpha_c)

alpha_c.pop(0)

letters.append(F)

G = [perceptron.PerceptronG(0.2) for i in range(19)]

add_name2(G, alpha_c)

alpha_c.pop(0)

letters.append(G)

H = [perceptron.PerceptronH(0.2) for i in range(18)]

add_name2(H, alpha_c)

alpha_c.pop(0)

letters.append(H)

I = [perceptron.PerceptronI(0.2) for i in range(17)]

add_name2(I, alpha_c)

alpha_c.pop(0)

letters.append(I)

J = [perceptron.PerceptronJ(0.2) for i in range(16)]

add_name2(J, alpha_c)

alpha_c.pop(0)

letters.append(J)

K = [perceptron.PerceptronK(0.2) for i in range(15)]

add_name2(K, alpha_c)

alpha_c.pop(0)

letters.append(K)

L = [perceptron.PerceptronL(0.2) for i in range(14)]

add_name2(L, alpha_c)

alpha_c.pop(0)

letters.append(L)

M = [perceptron.PerceptronM(0.2) for i in range(13)]

add_name2(M, alpha_c)

alpha_c.pop(0)

letters.append(M)

N = [perceptron.PerceptronN(0.2) for i in range(12)]

add_name2(N, alpha_c)

alpha_c.pop(0)

letters.append(N)

O = [perceptron.PerceptronO(0.2) for i in range(11)]

add_name2(O, alpha_c)

alpha_c.pop(0)

letters.append(O)

P = [perceptron.PerceptronP(0.2) for i in range(10)]

add_name2(P, alpha_c)

alpha_c.pop(0)

letters.append(P)

Q = [perceptron.PerceptronQ(0.2) for i in range(9)]

add_name2(Q, alpha_c)

alpha_c.pop(0)

letters.append(Q)

R = [perceptron.PerceptronR(0.2) for i in range(8)]

add_name2(R, alpha_c)

alpha_c.pop(0)

letters.append(R)

S = [perceptron.PerceptronS(0.2) for i in range(7)]

add_name2(S, alpha_c)

alpha_c.pop(0)

letters.append(S)

T = [perceptron.PerceptronT(0.2) for i in range(6)]

add_name2(T, alpha_c)

alpha_c.pop(0)

letters.append(T)

U = [perceptron.PerceptronU(0.2) for i in range(5)]

add_name2(U, alpha_c)

alpha_c.pop(0)

letters.append(U)

V = [perceptron.PerceptronV(0.2) for i in range(4)]

add_name2(V, alpha_c)

alpha_c.pop(0)

letters.append(V)

W = [perceptron.PerceptronW(0.2) for i in range(3)]

add_name2(W, alpha_c)

alpha_c.pop(0)

letters.append(W)

X = [perceptron.PerceptronX(0.2) for i in range(2)]

add_name2(X, alpha_c)

alpha_c.pop(0)

letters.append(X)

Y = [perceptron.PerceptronY(0.2) for i in range(1)]

add_name2(Y, alpha_c)

alpha_c.pop(0)

letters.append(Y)

for letter in letters:
    for a in minus_Z:
        if a == letter[0].name:
            root = './shuffled/' + a + '/'
            files = [f for f in listdir(root)]

            for file in files:

                name, ext = os.path.splitext(file)
                aFile = root + file

                while current_accuracy >= prev_accuracy and e_count < 100:
                    df = pd.read_csv(aFile, index_col=0, header=None)

                    af = df.copy()

                    af = af.set_index(af[1])

                    af = af.drop(1, axis=1)

                    for x in letter:
                        x_name = x.name + x.name2
                        if x_name in name:
                            for i, row in enumerate(af.values):
                                row = np.insert(row, 0, 1)
                                x.set_test_data(row, af.index[i])
                                if recal:
                                    x.reweight()
                                    recal = False
                                result = x.test()
                                total += 1
                                if result == x.target:
                                    correct += 1
                                else:
                                    error += 1
                                    x.reweight()
                                    recal = True
                        if total is not 0:
                            e_count += 1
                            shuffle_and_write(aFile, df)
                            prev_accuracy = current_accuracy
                            current_accuracy = m.ceil(float(correct) / total * 100)

                            if prev_accuracy > current_accuracy:
                                x.rollback()

                            total = error = correct = 0
                            recal = False

                current_accuracy = prev_accuracy = e_count = 0

tfile = './shuffled/test_data_shuffled.csv'
tf = pd.read_csv(tfile, index_col=0, header=None)

tcf = tf.copy()

tcf = tcf.set_index(tcf[1])

tcf = tcf.drop(1, axis=1)

y_true = tcf.index.values

name, ext = os.path.splitext(tfile)
for i, row in enumerate(tcf.values):
    row = np.insert(row, 0, 1)
    for letter in letters:
        for z in letter:
            z_name = z.name + z.name2
            z.set_test_data(row, tcf.index[i])
            result = z.test()
            if result == 1:
                votes[z.name] += 1
            else:
                votes[z.name2] += 1

        vote = max(votes.iteritems(), key=operator.itemgetter(1))[0]
        test_votes[vote] += 1
        votes = dict.fromkeys(alpha, 0)
    count = max(test_votes.iteritems(), key=operator.itemgetter(1))[0]
    y.append(count)
    c += 1
    print c

print 'Confusion Matrix:\n\n'

print y

y_actul = pd.Series(y_true, name='Actual')
y_pred = pd.Series(y, name='Predicted')

confusion = pd.crosstab(y_actul, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

print confusion
