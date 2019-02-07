import math
import random

class Neuron (object):
    def __init__(self, node_id, net_in):
        self.id = node_id
        self.in_w = net_in
        self.out_w = self.in_w
        self.weights = []

class simpleNeuralNetwork (object):
    def __init__(self, input_size=2, hidden_layer_size=2, output_size=1, bias=0.05, lr=0.5, activations=['relu', 'sigmoid']):
        self.inputLayerSize = input_size
        self.hiddenLayerSize = hidden_layer_size
        self.outputLayerSize = output_size

        self.bias = bias
        self.learning_rate = lr
        self.fun_activations = activations

        self.layers = []
        self.buildGraph()

        self.forward()

    def forward(self):
        for k in range(1, len(self.layers)):
            for j in range(len(self.layers[k])):
                for i in range(len(self.layers[k-1])):
                    W = self.layers[k][j].weights[i] * self.layers[k-1][i].out_w
                    self.layers[k][j].in_w += W
                self.layers[k][j].in_w += self.bias
                self.layers[k][j].out_w = self.activation(k - 1, self.layers[k][j].in_w)

    def backpropagation(self):
        pass

    def train(self, trainset, epochs=10):
        self.features = [row[:len(row)-1] for row in trainset]
        self.labels = [row[-1] for row in trainset]
        print("len feat: %d" % len(self.features))
        print("len lbs: %d" % len(self.labels))
        for e in range(1, epochs + 1):
            # print("Epoch %d --------------------- " % e)
            self.loss = 0
            for r in range(len(trainset)):
                if self.inputLayerSize == len(self.features[0]) and self.outputLayerSize == 1:
                    for j in range(len(self.layers[0])):
                        self.layers[0][j].in_w = self.features[r][j]

                    self.forward()
                    
                    y = self.labels[r]
                    yhat = self.layers[-1][0].out_w
                    self.loss = 0.5 * math.pow( y - yhat, 2)
                    # print("Target = %.5f, Prediction = %.5f, Loss = %.5f" % (y, yhat, self.loss))

                    self.backpropagation()

    def activation(self, layer_id, value):
        if self.fun_activations[layer_id] == 'relu':
            return self.relu(value)
        if self.fun_activations[layer_id] == 'sigmoid':
            return self.sigmoid(value)
        return value

    def relu(self, x):
        return max(0, x)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def buildGraph(self):
        node_index = 1
        input_layer = []
        for i in range(self.inputLayerSize):
            input_layer.append(Neuron(node_index, 0))
            node_index += 1
        
        hidden_layer = []
        for i in range(self.hiddenLayerSize):
            hidden_layer.append(Neuron(node_index, 0))
            node_index += 1
        
        output_layer = []
        for i in range(self.outputLayerSize):
            output_layer.append(Neuron(node_index, 0))
            node_index += 1
        
        self.layers.append(input_layer)
        self.layers.append(hidden_layer)
        self.layers.append(output_layer)

        for k in range(1, len(self.layers)):
            for j in range(len(self.layers[k])):
                for i in range(len(self.layers[k - 1])):
                    self.layers[k][j].weights.append( random.random()*0.5 )

    def track(self):
        print("Input  layer size: %d" % self.inputLayerSize)
        print("Hidden layer size: %d" % self.hiddenLayerSize)
        print("Output layer size: %d" % self.outputLayerSize)
        print("Overall bias: %.3f" % self.bias)
        for k in range(len(self.layers)):
            print("\n----- layer %d --------------------------" % k)
            for j in range(len(self.layers[k])):
                node = self.layers[k][j]
                print("\n Neuron_%d, in: %.3f, out: %.3f," % (node.id, node.in_w, node.out_w))
                print(" weights: ", node.weights)


nn = simpleNeuralNetwork()
nn.track()

xorset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
nn.train(xorset)

nn.track()