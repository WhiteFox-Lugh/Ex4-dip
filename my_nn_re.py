import numpy as np
import math
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from progressbar import ProgressBar

mndata = MNIST("/Users/lugh/Desktop/KUInfo/winter-CS3rd/le4-dip/works")

# settings
d = 28 * 28
img_div = 255
# img_len = 10000
mnist_data = 60000
c = 10
# eta is learning rate
eta = 0.01
# mid layer num
m = 30

batch_size = 1


# sigmoid function
def f_sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


# softmax function
def f_softmax(t):
    alpha = np.max(t)
    r = np.exp(t - alpha) / np.sum(np.exp(t - alpha))
    return r


# one-hot vector
def one_hot_vector(t):
    return np.identity(c)[t]


# input_layer : (1, batch_size * d) -> (d = 784, batch_size)
def input_layer(x):
    tmp = x.reshape(d, batch_size)
    return tmp / img_div


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


# main
if __name__ == "__main__":
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)

    # number input
    print("input an integer from 0 to 9999.")
    idx = int(sys.stdin.readline(), 10)

    # init weight and bias
    np.random.seed(334)
    w1 = np.random.normal(0, math.sqrt(1 / d), m * d)
    w1 = w1.reshape((m, d))
    w2 = np.random.normal(0, math.sqrt(1 / m), c * m)
    w2 = w2.reshape((c, m))
    b1 = np.random.normal(0, math.sqrt(1 / d), m)
    b1 = b1.reshape((m, 1))
    b2 = np.random.normal(0, math.sqrt(1 / m), c)
    b2 = b2.reshape((c, 1))

    # X[idx] (d)
    img = X[idx]
    print("X[idx] shape : {0} \n {1}".format(np.shape(img), img))

    # input_layer (
    output_input_layer = input_layer(img)
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

    plt.imshow(X[idx], cmap=cm.gray)
    plt.show()
    # print(Y[idx])
    print(np.argmax(output_output_layer))


    # Exercise 2.