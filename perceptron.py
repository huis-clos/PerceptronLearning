# Perceptron classes

import numpy as np


# Perceptron Super class
class Perceptron(object):
    name = ""
    name2 = ""
    prev_w = []
    w = []  # weights
    x = []  # inputs
    target_letter = ""
    target = 0
    vote = "None"
    decision = 0

    # constructor
    # input eta value for learning
    def __init__(self, eta):
        self.w = np.asarray([np.random.uniform(-1, 1) for i in range(17)])
        self.prev_w = np.copy(self.w)
        self.eta = eta

    # stores the inputs in a numpy array and sets the target
    # based on the known true letter
    def set_test_data(self, data, letter):
        self.x = np.asarray(data)
        self.target_letter = letter
        if self.target_letter == self.name:
            self.target = 1
        else:
            self.target = -1

    # Runs the test. Takes the sign of the dot product
    # of the weights and the input, sets the decision
    # and the vote based on the result and returns the
    # 1 or -1 based on whether the threshold was met
    def test(self):
        y = np.sign(np.dot(self.w, self.x))

        if y < 0:
            self.decision = result = -1
            self.vote = self.name2
        else:
            self.decision = result = 1
            self.vote = self.name

        return result

    # Reverts weights back to the previous set of weights if the current
    # weight's performance/accuracy is lower than the last epoch's accuracy
    def rollback(self):
        self.w = []
        self.w = np.copy(self.prev_w)

    # sets the second name in the perceptron for the other letter
    # INPUT: a String
    # OUTPUT: None
    def name_update(self, name2):
        self.name2 = name2

    # Recalculates the weights for a perceptron
    #
    def reweight(self):

        self.prev_w = np.copy(self.w)

        for idx, i in enumerate(self.w):
            self.w[idx] = i + self.eta * self.target * self.x[idx]


# perceptron derived classes for A, B, C etc.
class PerceptronA(Perceptron):
    name = 'A'


class PerceptronB(Perceptron):
    name = 'B'


class PerceptronC(Perceptron):
    name = 'C'


class PerceptronD(Perceptron):
    name = 'D'


class PerceptronE(Perceptron):
    name = 'E'


class PerceptronF(Perceptron):
    name = 'F'


class PerceptronG(Perceptron):
    name = 'G'


class PerceptronH(Perceptron):
    name = 'H'


class PerceptronI(Perceptron):
    name = 'I'


class PerceptronJ(Perceptron):
    name = 'J'


class PerceptronK(Perceptron):
    name = 'K'


class PerceptronL(Perceptron):
    name = 'L'


class PerceptronM(Perceptron):
    name = 'M'


class PerceptronN(Perceptron):
    name = 'N'


class PerceptronO(Perceptron):
    name = 'O'


class PerceptronP(Perceptron):
    name = 'P'


class PerceptronQ(Perceptron):
    name = 'Q'


class PerceptronR(Perceptron):
    name = 'R'


class PerceptronS(Perceptron):
    name = 'S'


class PerceptronT(Perceptron):
    name = 'T'


class PerceptronU(Perceptron):
    name = 'U'


class PerceptronV(Perceptron):
    name = 'V'


class PerceptronW(Perceptron):
    name = 'W'


class PerceptronX(Perceptron):
    name = 'X'


class PerceptronY(Perceptron):
    name = 'Y'


class PerceptronZ(Perceptron):
    name = 'Z'
