import math
from Layer import Layer


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def softmax(self, element):
        return math.exp(element)

    def softmaxComplete(self, dataIn):
        exponent = [list(map(self.softmax, i)) for i in dataIn]
        dataOut = []
        for i in range(len(dataIn)):
            row = []
            for j in range(len(dataIn[0])):
                row.append(math.exp(dataIn[i][j])/sum(exponent[i]))
            dataOut.append(row)
        return dataOut

    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        dataOut = self.softmaxComplete(dataIn)
        super().setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        data = self.getPrevOut()
        exponent = [list(map(self.softmax, i)) for i in data]
        dataOut = []
        size = len(data)*len(data[0])
        for h in range(len(data)):
            for i in range(size):
                row = []
                for j in range(size):
                    if i == j:
                        g = math.exp(data[h][j])/sum(exponent[h])
                        row.append(g*(1-g))
                    else:
                        g_i = math.exp(data[h][h])/sum(exponent[h])
                        g_j = math.exp(
                            data[h][j])/sum(exponent[h])
                        row.append(-(g_i*g_j))

                dataOut.append(row)
        return dataOut
