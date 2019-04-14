import sys
import numpy as np
from matplotlib import pyplot as plt
from adaline import Adaline

def get_data(file_add):
    """
    file_add: data file address
    returns a tupple having 4 elements.
    test data, validation data and their labels.
    all of them are numpy arrays.
    the order of items is as follows:
    (test data, validation data, test label, validation label)
    """
    dfile = open(file_add, 'r')
    data = []
    label = []
    count = 0
    is_val = False
    maxp = None
    for line in dfile:
        line = line.strip()
        if not line:
            continue
        if line == 'val':
           is_val = True
           continue
        d, lbl = line.split(':')
        lbl = int(lbl)
        label.append(lbl)
        tmp = d.split(',')
        tmp = np.array([float(x) for x in tmp])
        if maxp is None:
            maxp = np.copy(tmp)
        else:
            indexes = maxp < tmp
            maxp[indexes] = tmp[indexes]
        data.append(tmp)
        if not is_val:
            count += 1
    dfile.close()
    if maxp is None:
        return None
    # normalize data
    # ndata = []
    # for pnt in data:
    #     tmp = np.divide(pnt, maxp)
    #     ndata.append(tmp)


    half = count
    test = np.array(data[:half])
    test_label = np.array(label[:half])
    validate = np.array(data[half:])
    validate_label = np.array(label[half:])
    return (test, validate, test_label, validate_label)

def plot_data(data, label, ax):
    """
    only for 2d data!
    plot given data (e.g. test data)
    according to their labels.
    """
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    # print(data)
    for point, lbl in zip(data, label):
        if lbl == 0:
            x0.append(point[0])
            y0.append(point[1])
        else:
            x1.append(point[0])
            y1.append(point[1])
    ax.plot(x0, y0, 'bo')
    ax.plot(x1, y1, 'ro')

def plot_line(w, b, ax, x0=0, x1=1):
    """
    only for 2d data!
    plot learned line
    """
    y0 = -(w[0] * x0 + b) / w[1]
    y1 = -(w[0] * x1 + b) / w[1]
    ax.plot([x0, x1], [y0, y1])

def heavyside(x):
    if x > 0:
        return 1
    return 0

def sigmoid(x):
    val = 1 / (1 + np.exp(-x))
    return val

def main():
    if len(sys.argv) != 2:
        print('You should give path to the data file in the argument')
        return
    file_add = sys.argv[1]
    tst, val, tst_lbl, val_lbl = get_data(file_add)
    vec_size = len(tst[0])
    neuron = Adaline(vec_size, 0.1)
    # train
    lr_curve = []
    cutoff = 1000
    iters = 0
    test_data_size = len(tst_lbl)
    while cutoff > iters:
        iters += 1
        err = 0
        for x, y in zip(tst, tst_lbl):
            p = neuron.input(x)
            t = heavyside(p)
            if t != y:
                err += 1
        tmp = err / test_data_size
        lr_curve.append(tmp)
        for x, y in zip(tst, tst_lbl):
            p = neuron.input(x)
            # activation function
            t = sigmoid(p)
            # print('input:', x)
            # print('out:', p)
            # print('expected:', y)
            neuron.feedback(t, y, x)
        print(f"iteration {iters} finished)")
    print(f"learned weights: {neuron.w} bias: {neuron.b}")
    plt.figure()
    plt.plot(lr_curve)
    plt.show()
    # evaluate
    count = len(val)
    if count > 0:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for x, y in zip(val, val_lbl):
            p = neuron.input(x)
            t = heavyside(p)
            if t == y:
                if t == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if t == 1:
                    FP += 1
                else:
                    FN += 1
        if (TP + FP) != 0:
            percision = TP / (TP + FP)
            print(f"Percision: {percision:.2f}")
        if (TP + FN) != 0:
            recall = TP / (TP + FN)
            print(f"Recall: {recall:.2f}")
    # plot
    if vec_size == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_data(tst, tst_lbl, ax)
        x0 = np.min(tst, 0)[0]
        x1 = np.max(tst, 0)[0]
        plot_line(neuron.w, neuron.b, ax, x0, x1)
        plt.show()


if __name__ == '__main__':
    main()
