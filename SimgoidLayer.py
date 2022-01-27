import math
from Layer import Layer


class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def sigmoid(self, element):
        return 1/(1+math.exp(-element))

    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        dataOut = [list(map(self.sigmoid, i)) for i in dataIn]
        super().setPrevOut(dataOut)
        return dataOut

    def gradientHelper(self, element):
        return (self.sigmoid(element))*(1-self.sigmoid(element))

    def gradient(self):
        data = self.getPrevOut()
        response = [list(map(self.gradientHelper, i)) for i in data]
        return response
