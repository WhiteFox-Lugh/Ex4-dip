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
img_len = 10000
mnist_data = 60000
c = 10
# eta is learning rate
eta = 0.01
# mid layer num
# m = 528
m = 100

batch_size = 1
per_epoch = mnist_data // batch_size
epoch = 10


# input_layer : (1, batch_size * d) -> (d = 784, batch_size)
def input_layer(in_x):
    tmp = in_x.reshape(d, batch_size)
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


# calculate cross entropy
def cal_cross_entropy(prob, label):
    e = 0
    y_p = prob.T
    # y_p (100, 10)
    print(y_p)
    for j in range(batch_size):
        for k in range(c):
            e += (-label[j][k] * np.log(y_p[j][k]))
            # print("({0}, {1}) : {2}".format(j, k, e))

    return e


# main
if __name__ == "__main__":
    # loading data
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)
    p = ProgressBar()
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
    loss = np.array([])
    iteration = np.array([], dtype='int32')

    for itr in p(range(1000)):
        # random choice
        nums = list(range(0, X.size // d))
        choice_nums = np.random.choice(nums, batch_size, replace=False)
        input_img = np.array([], dtype='int32')
        t_label = np.array([], dtype='int32')
        t_label_one_hot = np.array([], dtype='int32')

        for i in range(batch_size):
            input_img = np.append(X[choice_nums[i]].reshape(d), input_img)
            t_label = np.append(Y[choice_nums[i]], t_label)

        # make one-hot vector
        t_label_one_hot = one_hot_vector(t_label)

        # 3-layer NN main
        # X1
        output_input_layer = input_layer(input_img)
        # X2
        output_mid_layer = mid_layer(w1, output_input_layer, b1)
        result = output_layer(w2, output_mid_layer, b2)
        y = []

        for i in range(batch_size):
            y += [np.argmax(result.T[i])]

        # find cross entropy
        entropy = cal_cross_entropy(result, t_label_one_hot)
        entropy_average = np.sum(entropy) / batch_size

        # Exercise 3:
        # back propagation

        # 1. softmax and cross entropy
        # result.T -> y_k(2), t_label_one_hot -> y_k
        # grad_en_ak shape: (c, batch_size)
        grad_en_ak = (result - t_label_one_hot.T) / batch_size

        # 2. find grad(E_n, X), grad(E_n, W2), grad(E_n, b2)
        # grad_en_x2: (m, c) * (c * batch_size) -> (m, batch_size)
        # grad_en_w2: (c * batch_size) * (batch_size * m) -> (c, m)
        grad_en_x2 = np.dot(w2.T, grad_en_ak)
        grad_en_w2 = np.dot(grad_en_ak, output_mid_layer.T)
        grad_en_b2 = np.array(np.sum(grad_en_ak, axis=1))

        # 3. back propagate : sigmoid
        bp_sigmoid = (1 - f_sigmoid(grad_en_x2)) * f_sigmoid(grad_en_x2)

        # 4. find grad(E_n, X), grad(E_n, W1), grad(E_n, b1)
        # grad_en_x1: (d, m) * (m * batch_size) -> (d, batch_size)
        # grad_en_w1: (m * batch_size) * (batch_size * d) -> (m, d)
        grad_en_x1 = np.dot(w1.T, bp_sigmoid)
        grad_en_w1 = np.dot(bp_sigmoid, output_input_layer.T)
        grad_en_b1 = np.array(np.sum(bp_sigmoid, axis=1))

        # 5. update parameters
        tmp_w1 = eta * grad_en_w1
        w1 = w1 - tmp_w1
        tmp_w2 = eta * grad_en_w2
        w2 = w2 - tmp_w2
        tmp_b1 = eta * grad_en_b1
        tmp_b1 = tmp_b1.reshape(m, 1)
        b1 = b1 - tmp_b1
        tmp_b2 = eta * grad_en_b2
        tmp_b2 = tmp_b2.reshape(c, 1)
        b2 = b2 - tmp_b2

        # 6. record entropy
        iteration = np.append(itr + 1, iteration)
        loss = np.append(entropy_average, loss)


    plt.plot(iteration, loss)
    plt.xlim([0, 1000])
    plt.ylim([0, 5])
    plt.title("entropy")
    plt.xlabel("itr")
    plt.ylabel("entropy")
    # plt.imshow(X[idx], cmap=cm.gray)
    plt.show()
    # print(Y[idx])

    # save parameter
    np.savez('test.npz', w1=w1, w2=w2, b1=b1, b2=b2)

