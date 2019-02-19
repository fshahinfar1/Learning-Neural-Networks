import numpy as np

class Perceptron:
    def __init__(self, n, lr=1):
        """
        n: vector size
        lr: learning rate (defualt: 1)
        """
        self.n = n
        self.lr = lr
        # initialize weights randomly
        self.w = np.random.rand(n)
        self.b = 0  # bias value
    
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
        self.check_train_data(x)
        y = np.dot(self.w, x) + self.b
        # print(y)
        if y > 0:
            return 1
        else:
            return 0

    def feedback(self, val, exp, x):
        """
        val: value input function returned
        exp: value that was expected (label)
        x: train data
        """
        self.check_train_data(x)
        has_err = False
        delta = 0
        if val == 1 and exp == 0:
            delta = -1
            has_err = True
        elif val == 0 and exp == 1:
            delta = 1
            has_err = True
        
        self.w = self.w + self.lr * delta * x
        self.b = self.b + self.lr * delta
        return has_err

