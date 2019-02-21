import sys
import numpy as np
from matplotlib import pyplot as plt
from perceptron import Perceptron

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
        tmp = [float(x) for x in tmp]
        data.append(tmp)
        if not is_val:
            count += 1
    dfile.close()
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


def main():
    file_add = sys.argv[1]
    tst, val, tst_lbl, val_lbl = get_data(file_add)
    vec_size = len(tst[0])
    neuron = Perceptron(vec_size, 0.1)
    # train
    has_err = True
    iters = 0
    cnt_err = 0
    while has_err:
        has_err = False
        cnt_err = 0
        iters += 1
        for x, y in zip(tst, tst_lbl):
            p = neuron.input(x)
            # print('input:', x)
            # print('out:', p)
            # print('expected:', y)
            b = neuron.feedback(p, y, x)
            if b:
                cnt_err += 1
            has_err = b or has_err
        print(f"iteration {iters} finished (err: {cnt_err})")
    print(f"learned weights: {neuron.w} bias: {neuron.b}")
    # evaluate
    count = len(val)
    if count > 0:
        cnt_err = 0
        for x, y in zip(val, val_lbl):
            p = neuron.input(x)
            if p != y:
                cnt_err += 1
        ratio = cnt_err / count
        print(f"err: {cnt_err}/{count} = {ratio:.2f}")
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
