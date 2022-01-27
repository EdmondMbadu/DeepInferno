
import csv
import math
from Layer import Layer
from FullyConnectedLayer import FullyConnectedLayer
from SimgoidLayer import SigmoidLayer
import numpy as np


class InputLayer(Layer):
    def __init__(self, dataIn):
        self.meanX = self.findMean(dataIn)
        self.stdX = self.findStandardDeviation(dataIn)

    def findMean(self, dataIn):
        mean_array = []
        for j in range(len(dataIn[0])):
            mean = 0.0
            for i in range(len(dataIn)):
                mean += dataIn[i][j]
            mean_array.append(round(mean/len(dataIn), 8))
        return mean_array

    def findStandardDeviation(self, dataIn):
        std_array = []
        for j in range(len(dataIn[0])):
            std = 0.0
            for i in range(len(dataIn)):
                std += math.pow(dataIn[i][j] - self.meanX[j], 2)
            std = round(math.sqrt(std/len(dataIn)), 8)
            std_array.append(1 if std == 0 else std)
        return std_array

    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        dataOut = []

        for i in range(len(dataIn)):
            row = []
            for j in range(len(dataIn[0])):
                row.append(round(
                    (dataIn[i][j]-self.meanX[j])/self.stdX[j], 8))
            dataOut.append(row)
        super().setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass


# with open("/Users/edmondmbadu/Documents/Drexel-Classes/Winter-quarter-2022/CS-615/CS615-H1-edmond-mbadu-enm58/deepThought/mcpd_augmented.csv") as file_name:
#     file_read = csv.reader(file_name)
#     array = list(file_read)

# arrayInput = [list(map(float, i))for i in array]
# # arrayInput = [[1, 2, 3, 4], [5, 6, 7, 8]]
# inputLayer = InputLayer(arrayInput)
# outPutOfInputLayer = inputLayer.forward(arrayInput)
# print("Output of input layer => ", np.array(outPutOfInputLayer))
# fullyConntected = FullyConnectedLayer(len(outPutOfInputLayer[0]), 2)
# outPutOfFullyConnectedLayer = fullyConntected.forward(outPutOfInputLayer)
# print("Output of fully connected layer => ",
#       np.array(outPutOfFullyConnectedLayer))
# sigmoid = SigmoidLayer()
# outPutOfSigmoid = sigmoid.forward(outPutOfFullyConnectedLayer)
# print("Output of sigmoid activation function => ", np.array(outPutOfSigmoid))
