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
# d_prime = 6
img_div = 255
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


# input_layer : (1, batch_size * d) -> (d = 784, batch_size)
def input_layer(x):
    # tmp = x.reshape(batch_size, d_prime)
    tmp = x.reshape(batch_size, d)
    return tmp.T / img_div


# mid_layer : (d = 784, batch_size) -> (m, batch_size)
def mid_layer(w, x, b):
    tmp = np.dot(w, x) + b
    # print("tmp mid layer: \n {0}".format(tmp))
    r = np.apply_along_axis(f_sigmoid, 0, tmp)
    # print("r_midlayer: \n {0}".format(r))
    return r


# output_layer : (m, batch_size) -> (c = 10, batch_size)
def output_layer(w, x, b):
    tmp = np.dot(w, x) + b
    # print("tmp output layer: \n {0}".format(tmp))
    r = np.apply_along_axis(f_softmax, 0, tmp)
    # print("r output layer: \n {0}".format(r))
    return r


# main
if __name__ == "__main__":
    # X, Y = mndata.load_training()
    X, Y = mndata.load_testing()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)

    # number input
    while True:
        print("input an integer from 0 to 9999.")
        idx = int(sys.stdin.readline(), 10)
        if 0 <= idx < 10000:
            break
        else:
            print("invalid input ;-(")

    # init weight and bias
    np.random.seed(3304)
    w1 = np.random.normal(0, math.sqrt(1 / d), m * d)
    w1 = w1.reshape((m, d))
    w2 = np.random.normal(0, math.sqrt(1 / m), c * m)
    w2 = w2.reshape((c, m))
    b1 = np.random.normal(0, math.sqrt(1 / d), m)
    b1 = b1.reshape((m, 1))
    b2 = np.random.normal(0, math.sqrt(1 / m), c)
    b2 = b2.reshape((c, 1))

    # random choice
    nums = list(range(0, X.size // d))

    # choice_nums = np.random.choice(nums, batch_size, replace=False)
    choice_nums = np.array([idx])
    input_img = np.array([], dtype='int32')
    t_label = np.array([], dtype='int32')
    t_label_one_hot = np.array([], dtype='int32')

    # -- test array for debug --
    """
    test = np.array([[[3, 3, 4], [4, 4, 5]], [[1, 2, 3], [4, 5, 6]], [[9, 9, 0], [8, 8, 7]]])
    w1_prime = np.array([2, 2, 3, 1, 1, 2, 1, 1, 2, 2, 1, 1])
    w1_prime = w1_prime.reshape((m, d_prime))
    w2_prime = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    w2_prime = w2_prime.reshape((c, m))
    b1_prime = np.array([1, 334])
    b1_prime = b1_prime.reshape((m, 1))
    b2_prime = np.array([4, 2, 1, 1, 2, 1, 1, 1, 1, 2])
    b2_prime = b2_prime.reshape((c, 1))
    """
    # -- test array end ---

    for i in range(batch_size):
        # input_img = np.append(X[choice_nums[i]], input_img)
        input_img = np.append(input_img, (X[choice_nums[i]]).reshape((d, 1)))
        # input_img = np.append(input_img, test[i].reshape((d_prime, 1)))
        # print("img {0} : \n {1}".format(i, input_img))
        # print("shape: {0}".format(np.shape(input_img)))
        t_label = np.append(Y[choice_nums[i]], t_label)

    # input_layer (
    output_input_layer = input_layer(input_img)
    output_mid_layer = mid_layer(w1, output_input_layer, b1)
    output_output_layer = output_layer(w2, output_mid_layer, b2)

    # print("debug softmax summation -> maybe printed [1] :-) ")
    # print(np.sum(output_output_layer, axis=0))

    # -- for debug area --
    # print("output_mid_layer is below")
    # print(output_mid_layer)
    # print("output_output_layer is below")
    # print(output_output_layer)
    # --- debug code area end ---

    for idx in range(batch_size):
        # plt.imshow(X[idx], cmap=cm.gray)
        plt.imshow(X[choice_nums[idx]], cmap=cm.gray)
        # print(t_label[idx])
        plt.show()

    # print(Y[idx])
    print(np.argmax(output_output_layer, axis=0))