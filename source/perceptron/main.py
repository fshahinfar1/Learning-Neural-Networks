import numpy as np
import sys
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

def main():
    file_add = sys.argv[1]
    tst, val, tst_lbl, val_lbl = get_data(file_add)
    vec_size = len(tst[0])
    neuron = Perceptron(vec_size, 0.1)
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


if __name__ == '__main__':
    main()
