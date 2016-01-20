#!/usr/bin/python
#
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
from pandas_confusion import ConfusionMatrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Adds the second name (the negative classification for AB perceptron (A is a positive target, B is a negative target)
# Input: an array of perceptron objects of the same primary name (A perceptrons, B perceptrons, etc), a letter
# Output: None
def add_name2(an_array, a):
    j = 0

    for i in a:
        an_array[j].name_update(i)
        j += 1

# reads in the appropriate data set, shuffles it, then writes back to the same file
# Input: a String, a  pandas dataframe
# Output: None
def shuffle_and_write(filename, df):

    for i in range(100):
        df = df.reindex(np.random.permutation(df.index))

    df.to_csv(filename, header=None)

# *******************************************************************************************
# ********* Main Function *******************************************************************
# *******************************************************************************************

e_count = 0 # epoch count
total = correct = error = t_correct = 0 # variable to hold accuracy calculation (number right, number, wrong, total samples
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
y_true = [] # for confusion matrix
y = []
votes = dict.fromkeys(alpha, 0) # votes from 325 perceptrons for a single test case
test_votes = dict.fromkeys(alpha, 0) #classifications for test cases

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

# *********************************************************************************************
# **************************** Begin Training *************************************************
# *********************************************************************************************

for letter in letters: # list of lists of perceptrons grouped by letter
    for a in minus_Z: # for folders in shuffled
        if a == letter[0].name: # to match on perceptron group
            root = './shuffled/' + a + '/'
            files = [f for f in listdir(root)] # files in shuffled/A directory, shuffled/B, etc

            for file in files:

                name, ext = os.path.splitext(file)
                aFile = root + file

                while current_accuracy >= prev_accuracy and e_count < 100:
                    #reads in and formats the csv into a dataframe
                    df = pd.read_csv(aFile, index_col=0, header=None)

                    af = df.copy()

                    af = af.set_index(af[1])

                    af = af.drop(1, axis=1)

                    # training on a perceptron in the group
                    for x in letter:
                        x_name = x.name + x.name2 # if the filename matches the perceptron, run training
                        if x_name in name:
                            for i, row in enumerate(af.values): # for each row of data (test case/training sample)
                                row = np.insert(row, 0, 1) # add one to beginning of row to match w0 and x0 for calc
                                x.set_test_data(row, af.index[i]) # store input in a perceptron
                                if recal: # if incorrectly classified
                                    x.reweight()
                                    recal = False
                                result = x.test() # calculate the result
                                total += 1
                                if result == x.target: # check result for accuracy
                                    correct += 1
                                else:
                                    error += 1
                                    x.reweight()
                                    recal = True
                        if total is not 0: # if we trained, increment epoch, check accuracy vs prev accuracy, shuffle data
                            e_count += 1
                            shuffle_and_write(aFile, df)
                            prev_accuracy = current_accuracy
                            current_accuracy = m.ceil(float(correct) / total * 100)

                            # print 'For ' + name + ' on perceptron' + x_name
                            # print '\tEpoch ', e_count
                            # print '\t\tCorrect: ', current_accuracy
                            # print '\t\tTotal samples: ', total
                            # print '\t\tError: ', m.ceil(float(error) / total * 100), '\n'

                            if prev_accuracy > current_accuracy: # call rollback on perceptron if accuracy degraded
                                x.rollback()

                            total = error = correct = 0
                            recal = False

                current_accuracy = prev_accuracy = e_count = 0


# ***************************************************************************************************
# ****************************** Begin Test *********************************************************
# ***************************************************************************************************

tfile = './shuffled/test_data_shuffled.csv'
tf = pd.read_csv(tfile, index_col=0, header=None)  # read in testing file to pandas dataframe

tcf = tf.copy() # begin formating for testing

tcf = tcf.set_index(tcf[1])

tcf = tcf.drop(1, axis=1)   # finish formating for testing

y_true = tcf.index.values # get correct letters for confusion matrix axis

name, ext = os.path.splitext(tfile)
for i, row in enumerate(tcf.values):
    row = np.insert(row, 0, 1) # for test case in dataset
    for letter in letters: # for group of perceptron from list of all perceptrons
        for z in letter: # for each perceptron in group
            z_name = z.name + z.name2
            z.set_test_data(row, tcf.index[i])
            result = z.test()
            if result == 1:
                votes[z.name] += 1 # insert vote into dictionary of key value pairs letter: count f
            else:
                votes[z.name2] += 1 # insert vote into dictionary of key value pairs letter: count f

    count = max(votes.iteritems(), key=operator.itemgetter(1))[0] # get most frequent letter
    votes = dict.fromkeys(alpha, 0) # reset dictionary for next test case
    y.append(count) # add classfication to array for confusion matrix
    if count == tcf.index[i]: #if the vote matches the known value of the target incremement correct
        t_correct += 1

# ***************************************************************************************************
# **************************** Creates and Displays Confusion Matrix ********************************
# ***************************************************************************************************

# uses pandas_confusion library to generate confusion matrix
print '\n\nConfusion Matrix:\n\n'

print '\tAccuracy is: ', m.ceil(float(t_correct) / 10000 * 100), '\n\n'


y_actul = pd.Series(y_true, name='Actual')
y_pred = pd.Series(y, name='Predicted')

confusion1 = ConfusionMatrix(y_actul, y_pred)
#
confusion1.print_stats()

confusion2 = pd.crosstab(y_actul, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

print confusion2

print confusion_matrix(y_actul, y_pred)



