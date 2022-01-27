

class LeastSquares:

    def eval(self, y, yhat):
        return (y-yhat)**2

    def gradient(self, y, yhat):
        return -2*(y-yhat)
