import numpy as np
import math
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

mndata = MNIST("/Users/lugh/Desktop/KUInfo/winter-CS3rd/le4-dip/works")

# settings
d = 3
img_len = 10000
c = 2
# mid layer num
m = 3
# load parameters
parameter = np.load('test.npz')
w1 = parameter['w1']
w2 = parameter['w2']
b1 = parameter['b1']
b2 = parameter['b2']


# input_layer
def input_layer(in_x):
    return in_x.reshape(d, 1)


# mid_layer : (d = 784, batch_size) -> (m = 30, batch_size)
def mid_layer(w, x, b):
    tmp = np.dot(w, x) + b
    r = np.apply_along_axis(f_sigmoid, 0, tmp)
    return r


# output_layer : (m = 30, batch_size) -> (c = 10, batch_size)
def output_layer(w, x, b):
    tmp = np.dot(w, x) + b
    r = np.apply_along_axis(f_softmax, 0, tmp)
    return r


# sigmoid function
def f_sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


# softmax function
def f_softmax(t):
    alpha = np.max(t)
    r = np.exp(t - alpha) / np.sum(np.exp(t - alpha))
    return r


if __name__ == "__main__":
    #X, Y = mndata.load_training()
    #X = np.array(X)
    #X = X.reshape((X.shape[0], 28, 28))
    #Y = np.array(Y)
    X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    Y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
    # number input
    print("input an integer from 0(000) to 7(111).")
    idx = int(sys.stdin.readline(), 10)
    # print (X[idx])
    output_input_layer = input_layer(X[idx])
    output_mid_layer = mid_layer(w1, output_input_layer, b1)
    output_output_layer = output_layer(w2, output_mid_layer, b2)
    print("debug softmax summation -> maybe printed [1] :-) ")
    print(np.sum(output_output_layer))
    # for debug area
    print("output_mid_layer is below")
    print(output_mid_layer)
    print("output_output_layer is below")
    print(output_output_layer)
    # --- debug code area end ---
    #plt.imshow(X[idx], cmap=cm.gray)
    #plt.show()
    # print(Y[idx])
    print(np.argmax(output_output_layer))