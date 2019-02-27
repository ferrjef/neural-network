import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

class SimpleNeuronNetwork:
    
    ''' A simple neuron network implementation '''
    
    def __init__(self, input_size=2, hidden_layer_size=3, output_size=1, learning_rate=0.5):
        self.layer_sizes = [input_size, hidden_layer_size, output_size]
        self.W = self.random_initial_weights()
        self.learning_rate = learning_rate

    def random_initial_weights(self):
        weights = []
        for i in range(1, len(self.layer_sizes)):
            weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]))
        return weights
    
    def forward(self, input):
        self.input = np.array(input)
        self.Z = []
        self.a = []

        self.Z.append(np.dot(self.input, self.W[0]))
        self.a.append(self.activation(self.Z[0]))
        for i in range(1, len(self.W)):
            self.Z.append(np.dot(self.a[i-1], self.W[i]))
            self.a.append(self.activation(self.Z[i]))
        
        # prediction (e.g yhat)
        return self.a[-1] 
    
    def backpropagation(self, y):
        self.yhat = self.forward(self.input)
        self.deltas = []
        self.J = []
    
        self.deltas.append( np.multiply(self.yhat - y, self.derivate(self.Z[-1] )) )
        self.J.append( np.dot(self.a[0].T, self.deltas[0]) )
        self.W[1] = self.W[1] - self.learning_rate * self.J[0]

        self.deltas.append( np.dot(self.deltas[-1], self.W[-1].T) * self.derivate(self.Z[0]) )
        self.J.append( np.dot(self.input.T, self.deltas[-1]) )
        self.W[0] = self.W[0] - self.learning_rate * self.J[1]
        
        return self.J

    def activation(self, z, fn="sigmoid"):
        if fn == "sigmoid":
            return 1 / (1 + np.exp(-z))
        return z
    
    def derivate(self, z, fn="sigmoid"):
        if fn == "sigmoid":
            return np.exp(-z) / ((1 + np.exp(-z))**2)
        return z
    
    def loss(self, y):
        yhat = self.a[-1]
        # total loss
        return 0.5 * sum((y - yhat)**2)


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

nn = SimpleNeuronNetwork(learning_rate=0.6)

losses = []

epochs = 8000
batch_size = 30
for epoch in range(epochs):
    pred = nn.forward(X)
    if epoch % batch_size == 0:
        loss = nn.loss(y)[0]
        losses.append(loss)
        print("Epoch {0}:\n\tLoss = {1:.5f}".format(epoch//batch_size, loss))
    nn.backpropagation(y)

pred = nn.forward(X)
print("Soft Target\n", pred)
print("Hard Target\n", np.round(pred))

plt.plot(range(len(losses)), losses)
plt.xlabel("epochs")
plt.ylabel("Loss")

plt.show()