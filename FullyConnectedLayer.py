
import random
from Layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        random.seed(0)
        self.weights = [[random.uniform(-.0001, .0001)
                         for x in range(sizeOut)] for y in range(sizeIn)]

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def getBias(self):
        return self.bias

    def sefBias(self, bias):
        self.bias = bias

    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        XW = np.dot(dataIn, self.weights)
        random.seed(0)
        self.bias = [[random.uniform(-.0001, .0001)
                      for j in range(len(XW[0]))] for i in range(len(XW))]
        Y = [[XW[i][j]+self.bias[i][j] for j in range(len(XW[0]))]
             for i in range(len(XW))]
        super().setPrevOut(Y)
        return Y

    def gradient(self):
        W_T = []
        W = self.getWeights()
        for j in range(len(W[0])):
            row = []
            for i in range(len(W)):
                row.append(W[i][j])
            W_T.append(row)
        return W_T
