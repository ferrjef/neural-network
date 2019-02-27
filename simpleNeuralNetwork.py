import math
import numpy as np

np.random.seed(7)

class SimpleNeuronNetwork:
    
    ''' A simple neuron network implementation '''
    
    def __init__(self, input_size=2, hidden_layers_sizes=[3, 3], output_size=1):
        self.layer_sizes = [input_size] + hidden_layers_sizes + [output_size]
        self.W = self.random_initial_weights()

    def random_initial_weights(self):
        weights = []
        for i in range(1, len(self.layer_sizes)):
            weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]))
        return weights
    
    def forward(self, inputs):
        self.inputs = inputs
        self.Z = []
        self.a = []

        self.Z.append(np.dot(self.inputs, self.W[0]))
        self.a.append(self.activation(self.Z[0]))
        for i in range(1, len(self.W)):
            self.Z.append(np.dot(self.a[i-1], self.W[i]))
            self.a.append(self.activation(self.Z[i]))
        
        return self.a[-1] # prediction (e.g yhat)
    
    def activation(self, z, fn="sigmoid"):
        if fn == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif fn == "softmax":
            return max(0, z)
        return z
    
    def derivate(self, z, fn="sigmoid"):
        if fn == "sigmoid":
            return np.exp(-z) / ((1 + np.exp(-z))**2)
        elif fn == "softmax":
            return 0 if z <= 0 else 1
        return z
    
    def backpropagation(self, y):
        self.yhat = self.forward(self.inputs)
        self.deltas = []
        self.J = []
    
        self.deltas.append( np.multiply(self.yhat - y, self.derivate(self.Z[-1])) )
        self.J.append( np.dot(self.a[0].T, self.deltas[0]) )
    
        self.deltas.append( np.dot(self.deltas[-1], self.W[-1].T) * self.derivate(self.Z[0]) )
        self.J.append( np.dot(self.inputs.T, self.deltas[-1]) )
        
        return self.J
    
    def loss(self, y):
        yhat = self.a[-1]
        return 0.5 * sum((y - yhat)**2)