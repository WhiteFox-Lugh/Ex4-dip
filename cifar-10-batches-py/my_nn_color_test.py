import numpy as np
from numpy import ndarray
import math
import sys
import matplotlib.pyplot as plt
from pylab import cm
import cPickle as cPickle


def unpickle(f):
    with open(f, 'rb') as fo:
        dict = cPickle.load(fo)
    d_x = np.array(dict['data'])
    d_x = d_x.reshape((d_x.shape[0],3,32,32))
    d_y = np.array(dict['labels'])
    return d_x, d_y


def main():
    data_x, data_y = unpickle("./data_batch_1")
    idx = 1000
    plt.imshow(data_x[idx].transpose((1, 2, 0)))
    plt.show()
    print(data_y[idx])


if __name__ == "__main__":
    main()
