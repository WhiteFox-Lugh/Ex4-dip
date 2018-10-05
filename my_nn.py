import numpy as np
import math
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

mndata = MNIST("/Users/lugh/Desktop/KUInfo/winter-CS3rd/le4-dip/works")

# settings
d = 28 * 28
img_len = 10000
c = 10
# mid layer num
m = 30


# init weight and bias
np.random.seed(334)
w1 = np.random.normal(0, math.sqrt(1 / d), m * d)
w1 = w1.reshape((m, d))
w2 = np.random.normal(0, math.sqrt(1 / m), c * m)
w2 = w2.reshape((c, m))
b1 = np.random.normal(0, math.sqrt(1 / d), m)
b2 = np.random.normal(0, math.sqrt(1 / m), c)


# input_layer
def input_layer(in_x):
    return in_x.reshape(d)


# mid_layer
def mid_layer(in_x):
    tmp = np.dot(w1, in_x) + b1
    result = f_sigmoid(tmp)
    return result


# output_layer
def output_layer(in_x):
    tmp = np.dot(w2, in_x) + b2
    result = f_softmax(tmp)
    return result


# sigmoid function
def f_sigmoid(t):
    return 1 / (1 + np.exp((-1) * t))


# softmax function
def f_softmax(t):
    alpha = np.max(t)
    result = np.exp(t - alpha) / np.sum(np.exp(t - alpha))
    return result


if __name__ == "__main__":
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)

    # number input
    print("input an integer from 0 to 9999.")
    idx = int(sys.stdin.readline(), 10)

    # print (X[idx])
    output_input_layer = input_layer(X[idx])
    output_mid_layer = mid_layer(output_input_layer)
    output_output_layer = output_layer(output_mid_layer)
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
