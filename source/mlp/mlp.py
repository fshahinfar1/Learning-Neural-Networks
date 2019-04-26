import numpy as np
from random import random

class MLP:
    def __init__(self, learning_rate):
        self.num_layers = 0
        self.input_size = 0
        self.has_input = False
        self.num_neurons_in_layer = []
        self.weights = []
        self.learning_rate = learning_rate
        self.sigmoid_parameter = 1
        self.last_calculation_res = None

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
    
    def activation_function(self, x):
        """
        AF = sigmoid
        assuming `x` is an vector
        return a vector the same size as x
        """
        a = self.sigmoid_parameter
        size = x.size
        value = np.zeros(size)
        for i in range(size):
            value[i] = 1 / (1 + exp(-a * x[i]))
        return value
    
    def activation_function_derivation(self, level):
        """
        AF = sigmoid
        calculate derivation of AF for asked layer
        returns a vector
        """
        a = self.sigmoid_parameter
        y = self.last_calculation_res[level]
        Q_prime = a * y * (1 - y)
        return Q_prime

    def input(self, vector):
        self.last_calculation_res.clear()
        last_layer = np.append(1, vector)  # add bias node
        for mat in self.weights:
            last_layer = np.dot(last_layer, mat)
            last_layer = self.activation_function(last_layer)
            self.last_calculation_res.append(last_layer)
        return last_layer
    
    def backpropagate(self, desired):
        if self.last_calculation_res is None:
            print('No input values yet')
            return
        
        a = self.sigmoid_parameter
        lr = self.learning_rate
        w = self.weights
        new_weights = []

        e = desired - output
        # activation function derivation
        Q_prime = self.activation_function_derivation(-1)
        local_derivation = e * Q_prime
        y = self.last_calculation_res[-1]
        delta_w = lr * local_derivation * y

        updated_w = w[-1] + delta_w
        new_weights.insert(0, updated_w)

        start_index = self.num_layers - 2
        for i in range(start_index, -1, -1):
            Q_prime = self.activation_function_derivation(i)
            local_derivation = Q_prime * np.dot(local_derivation, w[i])
            y = self.last_calculation_res[i]
            delta_w = lr * local_derivation * y
            updated_w = w[i] + delta_w
            new_weights.insert(0, updated_w)


        



