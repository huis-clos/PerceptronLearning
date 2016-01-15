#Perceptron class

import numpy as np

class Perceptron(object):

    def __init__(self, eta):
        self.w = np.asarray([np.random.uniform(-1, 1) for i in range(17)])
        self.eta = eta

    def set_test_data(self, data):
        self.x = np.asarray(data)

