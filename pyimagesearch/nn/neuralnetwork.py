import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the list of weight matrics,
        # then store the network architecture
        # and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers)-2):
            # randomly initialize a weight matrix
            # connecting the number of nodes in
            # each respective layer together, adding
            # an extra node for the bias
