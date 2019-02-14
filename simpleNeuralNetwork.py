import math
import random

class Neuron (object):
    def __init__(self, node_id, net_in):
        self.id = node_id
        self.in_w = net_in
        self.out_w = self.in_w
        self.weights = []
        self.loss = .0
        self.target = None

class simpleNeuralNetwork (object):
    def __init__(self, input_size=2, hidden_layer_size=2, output_size=2, bias=0.05, lr=0.5, activations=['relu', 'sigmoid']):
        self.inputLayerSize = input_size
        self.hiddenLayerSize = hidden_layer_size
        self.outputLayerSize = output_size

        self.bias = bias
        self.learning_rate = lr
        self.fun_activations = activations
        self.total_loss = 0.0

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

        print("Total loss: %.3f" % self.total_loss)

        for k in range(len(self.layers)-1,-1,-1):
            for j in range(len(self.layers[k])):
                if k == len(self.layers) - 1:
                    for i in range(len(self.layers[k-1])):
                        dLdOut = self.layers[k][j].out_w - self.layers[k][j].target
                        dOutdIn = self.sigmoid(self.layers[k][j].out_w)*(1 - self.sigmoid(self.layers[k][j].out_w))
                        dIndW = self.layers[k-1][i].out_w
                        self.layers[k][j].weights[i] = dLdOut * dOutdIn * dIndW
                else:
                    pass
                print("--> ", self.layers[k][j].weights)


        

    def train(self, trainset, epochs=10):
        self.features = [row[:len(trainset)-2] for row in trainset]
        for e in range(1, epochs + 1):
            # print("Epoch %d --------------------- " % e)
            self.loss = 0
            for r in range(len(trainset)):
                if self.inputLayerSize == len(self.features[0]):
                    for j in range(len(self.layers[0])):
                        self.layers[0][j].in_w = self.features[r][j]
                    self.forward()

                y = trainset[r][-1]
                for i in range(len(self.layers[-1])):
                    yhat = self.layers[-1][i].out_w
                    self.layers[-1][i].loss = 0.5 * math.pow( y - yhat , 2 )
                    self.layers[-1][i].target = y
                    self.total_loss += self.layers[-1][i].loss
                    #print("y = %.3f, yhat = %.3f, loss = %.3f" % (y, yhat, self.layers[-1][i].loss))
                
                self.backpropagation()
                self.total_loss = 0.0

    def activation(self, layer_id, value):
        if self.fun_activations[layer_id] == 'relu':
            return self.relu(value)
        if self.fun_activations[layer_id] == 'sigmoid':
            return self.sigmoid(value)
        return value

    def derivate(self, layer_id, y, x):
        if self.fun_activations[layer_id] == 'relu':
            return 0 if x <= 0 else 1
        if self.fun_activations[layer_id] == 'sigmoid':
            return x - y
        return 0

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


# sample

nn = simpleNeuralNetwork()
nn.track()

xorset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
nn.train(xorset,1)