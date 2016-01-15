import pandas as pd
import perceptron
import numpy as np
from sklearn import preprocessing

error = 0

df = pd.read_csv('letter-recognition.data', index_col = 0, header = None, names =
['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'])

df.index.name = 'letter'

df = df.div(15)

train, test = np.split(df, 2)

length, width = train.shape

A = [perceptron.Perceptron(0.2) for i in range(25)]

for i, row in enumerate(train.values):
    for x in A:
        x.set_test_data(row)
    break












