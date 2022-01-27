from cmath import tan, tanh
import math
from Layer import Layer


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def tanh(self, element):
        return (math.exp(element) - math.exp(-element))/(math.exp(element) + math.exp(-element))

    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        dataOut = [list(map(self.tanh, i)) for i in dataIn]
        super().setPrevOut(dataOut)
        return dataOut

    def gradientHelper(self, element):
        return 1 - (self.tanh(element)**2)

    def gradient(self):
        data = self.getPrevOut()
        response = [list(map(self.gradientHelper, i)) for i in data]
        return response
