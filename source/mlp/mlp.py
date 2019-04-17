import numpy as np
from random import random

class MLP:
    def __init__(self):
        self.num_layers = 0
        self.input_size = 0
        self.has_input = False
        self.num_neurons_in_layer = []
        self.weights = []

    def add_layer(self, num_neurons, is_input=False):
        if is_input:
            if has_input:
                raise Exception('Input layer was defined before')
            has_input = True
            input_size = num_neurons
        self.num_layers += 1
        self.num_neurons_in_layer.append(num_neurons)

    def setup_network(self):
        weights = np.array(())
        for layer in range(1, self.num_layers):
            last_layer_size = self.num_neurons_in_layer[layer - 1]
            this_layer_size = self.num_neurons_in_layer[layer]
            layer_mat = np.array((last_layer_size, this_layer_size))
            for neurons in range(this_layer_size):
                # initialize with random values
                # bias is included
                for i in range(last_layer_size):
                    layer_mat[i, layer - 1] = random()
            self.weights.append(layer_mat)

    def input(self, vector):
        pass

