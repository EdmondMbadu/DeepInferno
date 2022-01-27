import math
from Layer import Layer


class ReluLayer(Layer):
    def __init__(self):
        super().__init__()

    def findMax(self, element):
        return max(0, element)

    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        dataOut = [list(map(self.findMax, i)) for i in dataIn]
        super().setPrevOut(dataOut)
        return dataOut

    def gradientHelper(self, element):
        return 1 if element >= 0 else 0

    def gradient(self):
        data = self.getPrevOut()
        response = [list(map(self.gradientHelper, i)) for i in data]
        return response
