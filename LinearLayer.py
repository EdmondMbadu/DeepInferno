
from Layer import Layer


class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        dataOut = dataIn
        super().setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass
