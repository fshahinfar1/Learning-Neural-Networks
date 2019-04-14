import numpy as np

class Adaline:
    def __init__(self, n, lr=1):
        """
        n: vector size
        lr: learning rate (defualt: 1)
        """
        self.n = n
        self.lr = lr
        # initialize weights randomly
        self.w = np.random.rand(n)
        self.b = np.random.rand()  # bias value
    
    def check_train_data(self, x):
        """
        check if input vector has the same shape
        as expected.
        """
        shape = np.shape(x)
        if len(shape) != 1:
            raise Exception("Shape error")
        if shape[0] != self.n:
            raise Exception("Length error")


    def input(self, x):
        """
        calculate neuron output
        x: input vector
        """
        self.check_train_data(x)
        y = np.dot(self.w, x) + self.b
        print(x, y)
        return y

    def feedback(self, val, exp, x):
        """
        Gradiant Descent
        update weights and bias value. (learn)
        val: value input function returned
        exp: value that was expected (label)
        x: train data
        """
        self.check_train_data(x)
        delta = (exp - val)
        tmp = self.lr * delta * x
        # print(exp, val, delta)
        # print(x)
        # print(tmp)
        # print('====')
        self.w = self.w + tmp
        self.b = self.b + self.lr * delta

