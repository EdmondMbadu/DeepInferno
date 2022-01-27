

import math


class LogLoss:

    def eval(self, y, yhat):
        return -(y*math.log(yhat)+(1-y)*math.log(1-yhat))

    def gradient(self, y, yhat):
        return -(y-yhat)/(yhat*(1-yhat))
