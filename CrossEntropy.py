

import math


class CrossEntropy:

    def eval(self, y, yhat):
        sum = 0
        for i in range(len(y)):
            sum = sum + -y[i]*math.log(yhat[i])
        return sum

    def gradient(self, y, yhat):
        solution = []
        for i in range(len(y)):
            solution.append(-(y[i]/yhat[i]))
        return solution
