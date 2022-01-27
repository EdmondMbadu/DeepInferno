

from CrossEntropy import CrossEntropy
from InputLayer import InputLayer
from FullyConnectedLayer import FullyConnectedLayer
from LeastSquares import LeastSquares
from LogLoss import LogLoss
from ReluLayer import ReluLayer
from SimgoidLayer import SigmoidLayer
from SoftmaxLayer import SoftmaxLayer
from TanhLayer import TanhLayer
import numpy as np

# # Part 4 Testing gradient ( in order)
print("Part IV: Testing Gradient")
print("=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>")
arrayInput = np.array([[1, 2, 3, 4]])
print("Input X for testing Layers: ", arrayInput)
print("==========================================================")
inputLayer = InputLayer(arrayInput)
print("Output of input layer:", inputLayer.forward(arrayInput), "\n")

fullyConnectedLayer = FullyConnectedLayer(len(arrayInput[0]), 2)
print("Output of fully Connected layer",
      fullyConnectedLayer.forward(arrayInput), "\n")
print("Gradient of fully connected layer",
      fullyConnectedLayer.gradient(), "\n")

relu = ReluLayer()
print("Output of Relu activation layer: ", relu.forward(arrayInput), "\n")
print("Gradient of relu layer: ",
      relu.gradient(), "\n")


sigmoid = SigmoidLayer()
print("Output of simgoid activation layer: ",
      sigmoid.forward(arrayInput), "\n")
print("Gradient of Sigmoid Layer: ", sigmoid.gradient(), "\n")

softmax = SoftmaxLayer()
print("Output of softmax activation layer: ",
      softmax.forward(arrayInput), "\n")
print("Gradient of Softmax layer: ", softmax.gradient(), "\n")

tanh = TanhLayer()
print("Output of hyperbolic tangent layer: ", tanh.forward(arrayInput), "\n")
print("Gradient of Tanh layer: ", tanh.gradient(), "\n")


# Part 5
print("Part V: Testing the Objective Layers")
print("=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>")
least = LeastSquares()
print("Least Square Objective function with y = 0 and yhat = .2 : ", least.eval(0, .2))
print("Least Square Graduent with y = 0 and  yhat= .2", least.gradient(0, .2))

log = LogLoss()
print("Objective function with y = 0 and yhat = .2: ", log.eval(0, .2))
print("Log Loss gradient with y = 0 and yhat = .2", log.gradient(0, .2))
cross = CrossEntropy()
print("Cross entropy for the given inputs:",
      cross.eval([1, 0, 0], [.2, .2, .6]))
print("Cross entropy gradient for the given input",
      cross.gradient([1, 0, 0], [.2, .2, .6]))
