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
c = 10
# eta is learning rate
eta = 0.01
# mid layer num
m = 200
batch_size = 100
per_epoch = 60000 // batch_size
epoch = 15

# init weight and bias
np.random.seed(3304)
# global w1
w1 = np.random.normal(0, math.sqrt(1 / d), m * d)
w1 = w1.reshape((m, d))
# global w2
w2 = np.random.normal(0, math.sqrt(1 / m), c * m)
w2 = w2.reshape((c, m))
# global b1
b1 = np.random.normal(0, math.sqrt(1 / d), m)
b1 = b1.reshape((m, 1))
# global b2
b2 = np.random.normal(0, math.sqrt(1 / m), c)
b2 = b2.reshape((c, 1))
loss = np.array([])
iteration = np.array([], dtype='int32')
# p = ProgressBar()

"""
class GD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grad):
        for k in params.keys():
            params[k] -= self.lr * grad[k]
"""

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
    tmp = x.reshape(batch_size, d)
    # tmp = x.reshape(batch_size, d)
    return tmp.T / img_div


# mid_layer : (d = 784, batch_size) -> (m, batch_size)
def mid_layer(w, x, b):
    return np.dot(w, x) + b


# mid_layer_activation_function
def mid_layer_activation(t):
    return np.apply_along_axis(f_sigmoid, axis=0, arr=t)


# output_layer : (m, batch_size) -> (c = 10, batch_size)
def output_layer(w, x, b):
    return np.dot(w, x) + b


# output_layer_activation_function
def output_layer_activation(t):
    return np.apply_along_axis(f_softmax, axis=0, arr=t)


# one-hot vector
def one_hot_vector(t):
    return np.identity(c)[t]


# calculate cross entropy
def cal_cross_entropy(prob, label):
    e = np.array([], dtype="float32")
    y_p = prob.T
    # y_p (100, 10)
    # print(y_p)
    for j in range(batch_size):
        tmp_e = 0
        for k in range(c):
            tmp_e += (-label[j][k] * np.log(y_p[j][k]))
            # print("({0}, {1}) : {2}".format(j, k, e))
        e = np.append(e, tmp_e)
    return e


# main
if __name__ == "__main__":
    X, Y = mndata.load_training()
    # X, Y = mndata.load_testing()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)

    # --- number input ---
    """
    while True:
        print("input an integer from 0 to 9999.")
        idx = int(sys.stdin.readline(), 10)
        if 0 <= idx < 10000:
            break
        else:
            print("invalid input ;-(")
    """

    # random choice
    nums = list(range(0, X.size // d))

    for itr in range(epoch * per_epoch):
        # init
        choice_nums = np.array([], dtype='int32')
        input_img = np.array([], dtype='int32')
        t_label = np.array([], dtype='int32')
        t_label_one_hot = np.array([], dtype='int32')
        output_input_layer = np.array([], dtype='float64')
        a_mid_layer = np.array([], dtype='float64')
        z_mid__layer = np.array([], dtype='float64')
        a_output_layer = np.array([], dtype='float64')
        result = np.array([], dtype='float64')

        choice_nums = np.random.choice(nums, batch_size, replace=False)
        # choice_nums = np.array([idx])

        """
        print("choice nums: \n {0}".format(choice_nums))
        print("input: \n {0}".format(input_img))
        print("t label: \n {0}".format(t_label))
        print("one: \n {0}".format(t_label_one_hot))
        print("shape: {0}, {1}, {2}, {3}".format(np.shape(choice_nums), np.shape(input_img), np.shape(t_label), np.shape(t_label_one_hot)))
        """

        for i in range(batch_size):
            tmp_img = X[choice_nums[i]]
            input_img = np.append(input_img, tmp_img)
            t_label = np.append(t_label, Y[choice_nums[i]])


        # make one-hot vector
        t_label_one_hot = one_hot_vector(t_label)

        # NN
        # input_layer : (1, batch_size * d) -> (d = 784, batch_size)
        output_input_layer = input_layer(input_img)
        # mid_layer : (d = 784, batch_size) -> (m, batch_size)
        a_mid_layer = mid_layer(w1, output_input_layer, b1)
        z_mid_layer = mid_layer_activation(a_mid_layer)
        # output_layer : (m, batch_size) -> (c = 10, batch_size)
        a_output_layer = output_layer(w2, z_mid_layer, b2)
        result = output_layer_activation(a_output_layer)

        # find cross entropy
        entropy = cal_cross_entropy(result, t_label_one_hot)
        entropy_average = np.sum(entropy, axis=0) / batch_size

        # Exercise 3: back propagation

        # 1. softmax and cross entropy
        # result.T -> y_k(2), t_label_one_hot -> y_k
        # grad_en_ak shape: (c, batch_size)
        grad_en_ak = (result - t_label_one_hot.T) / batch_size

        # 2. find grad(E_n, X), grad(E_n, W2), grad(E_n, b2)
        # grad_en_x2: (m, c) * (c * batch_size) -> (m, batch_size)
        # grad_en_w2: (c * batch_size) * (batch_size * m) -> (c, m)
        grad_en_x2 = np.dot(w2.T, grad_en_ak)
        grad_en_w2 = np.dot(grad_en_ak, z_mid_layer.T)
        grad_en_b2 = grad_en_ak.sum(axis=1)
        grad_en_b2 = grad_en_b2.reshape((c, 1))

        # 3. back propagate : sigmoid
        # bp_sigmoid -> (m, batch_size)
        bp_sigmoid = np.dot(w2.T, grad_en_ak) * (1 - f_sigmoid(a_mid_layer)) * f_sigmoid(a_mid_layer)

        # 4. find grad(E_n, X), grad(E_n, W1), grad(E_n, b1)
        # grad_en_x1: (d, m) * (m * batch_size) -> (d, batch_size)
        # grad_en_w1: (m * batch_size) * (batch_size * d) -> (m, d)
        grad_en_x1 = np.dot(w1.T, bp_sigmoid)
        grad_en_w1 = np.dot(bp_sigmoid, output_input_layer.T)
        grad_en_b1 = bp_sigmoid.sum(axis=1)
        grad_en_b1 = grad_en_b1.reshape((m, 1))

        # 5. update parameter
        w1 -= eta * grad_en_w1
        w2 -= eta * grad_en_w2
        b1 -= eta * grad_en_b1
        b2 -= eta * grad_en_b2
        # print("b2 is: \n {0}".format(b2))

        # 6. record entropy
        iteration = np.append(iteration, itr + 1)
        loss = np.append(loss, entropy_average)
        # print("itr is: {0}, loss is: {1} \n ".format(itr + 1, entropy_average))

        if itr % per_epoch == per_epoch - 1:
            # print("loss: \n {0}".format(loss))
            plt.plot(iteration, loss)
            plt.title("entropy")
            plt.xlabel("itr")
            plt.ylabel("entropy")
            plt.show()

    # -- for showing images --
    """
    for idx in range(batch_size):
        # plt.imshow(X[idx], cmap=cm.gray)
        plt.imshow(X[choice_nums[idx]], cmap=cm.gray)
        # print(t_label[idx])
        plt.show()
    """
    # -- end --

    # print(Y[idx])
    # print(np.argmax(result, axis=0))