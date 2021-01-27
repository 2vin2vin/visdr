#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module Name:        AirodynCV
#   Submodule Name:     networks
#   File Name:          nn.py
#   Author:             Aviral Sharma
# ______________________________________

# importing necessary packages
import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        """Class for Artificial Neural Network, using back-propagation"""
        # initialize the weight list, network architecture, learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of first 2 layers, stop before reaching the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initializing weights matrix, adding extra node for bias trick

            # if layers[i] = 2 (2 nodes in ith layer) and layers[i+1]=2 then weight matrix is 2x2 since all the
            # nodes in ith layer must be connected to all the nodes in (i+1)th layer other than the bias node.
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)

            # scaling by sqrt of number of nodes in current layer, normalizing the variance of each neuron's output
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append((w / np.sqrt(layers[-2])))

    def __repr__(self):
        # return string that will print the neural network architecture
        return "Neural Network Architecture: {}".format("-".join(str(j) for j in self.layers))

    def sigmoid(self, x):
        """Sigmoid Activation Function
        sigma(x) = 1.0 / (1 + e^(-x))"""
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        """Derivative of the Sigmoid Activation Function\n
        used during backward pass\n
        Computes the derivative assuming that 'x' has already passed through the 'sigmoid' function"""
        return x * (1 - x)

    def fit(self, X, y, epochs=100, displayUpdate=100):
        """Fit function is inspired from the sk-learn library.
        X  = training data\n
        y = ground-truth class labels for X\n
        epochs = number of epochs to train the data for\n
        displayUpdate = number of epochs to let pass before printing training update"""
        X = np.c_[X, np.ones((X.shape[0]))]  # concatenation of bias trick ones array to X feature

        # looping over epochs k_number=32, k_size=(3, 3)
        for epoch in np.arange(0, epochs):
            # looping over individual feature vector
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

        # check if the training display requires update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[UPDATE] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        """Back Propagation\n
        x: Individual feature vector from design matrix\n
        y: Corresponding ground-truth label"""

        # list A stores output activation from activation function for each layer as data point 'x' forward propagates
        # through the network.
        a = [np.atleast_2d(x)]

        # Forward Propagation
        for layer in np.arange(0, len(self.W)):
            net = a[layer].dot(self.W[layer])

            out = self.sigmoid(net)

            a.append(out)

        # Back Propagation
        error = a[-1] - y

        # If the sigmoid functions gives a HIGH or LOW value(Pretty good confidence), the derivative of that value
        # is LOW. If it gives a value at the steepest slope(0.5), the derivative of that value is HIGH.
        # When the function gives a bad prediction, we want to change our weights by a higher number,
        # and on the contrary, if the prediction is good(High confidence), we do NOT want to change our weights much.
        # CHECK THE CURVE OF SIGMOID FUNCTION AND ITS DERIVATIVE
        d = [error * self.sigmoid_deriv(a[-1])]

        for layer in np.arange(len(a) - 2, 0, -1):
            delta = d[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(a[layer])
            d.append(delta)

        d = d[::-1]

        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * a[layer].T.dot(d[layer])

    def predict(self, X, addBias=True):

        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):

        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
# _________________________________________________ END OF FILE __________________________________________________ #
