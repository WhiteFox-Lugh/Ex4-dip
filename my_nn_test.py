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
batch_size = 1
per_epoch = 60000 // batch_size
epoch = 15

# init vars
# np.random.seed(3304)
# X, Y = mndata.load_training()
X, Y = mndata.load_testing()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28))
Y = np.array(Y)

# for Exercise 1
network = {}

# for Exercise 4
# network = np.load(params.npz)
nums = list(range(0, X.size // d))

w1_tmp = np.random.normal(0, math.sqrt(1 / d), m * d)
network['w1'] = w1_tmp.reshape((m, d))
w2_tmp = np.random.normal(0, math.sqrt(1 / m), c * m)
network['w2'] = w2_tmp.reshape((c, m))
b1_tmp = np.random.normal(0, math.sqrt(1 / d), m)
network['b1'] = b1_tmp.reshape((m, 1))
b2_tmp = np.random.normal(0, math.sqrt(1 / m), c)
network['b2'] = b2_tmp.reshape((c, 1))


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
    for j in range(batch_size):
        tmp_e = 0
        for k in range(c):
            tmp_e += (-label[j][k] * np.log(y_p[j][k]))
        e = np.append(e, tmp_e)
    return e


# forwarding
def forward():
    data_forward = {}

    # input_layer : (1, batch_size * d) -> (d = 784, batch_size)
    output_input_layer = input_layer(input_img)
    # mid_layer : (d = 784, batch_size) -> (m, batch_size)
    a_mid_layer = mid_layer(network['w1'], output_input_layer, network['b1'])
    z_mid_layer = mid_layer_activation(a_mid_layer)
    # output_layer : (m, batch_size) -> (c = 10, batch_size)
    a_output_layer = output_layer(network['w2'], z_mid_layer, network['b2'])
    result = output_layer_activation(a_output_layer)

    # find cross entropy
    # entropy = cal_cross_entropy(result, network['t_label_one_hot'])
    # entropy_average = np.sum(entropy, axis=0) / batch_size

    data_forward['x1'] = output_input_layer
    data_forward['a1'] = a_mid_layer
    data_forward['z1'] = z_mid_layer
    data_forward['a2'] = a_output_layer
    data_forward['y'] = result
    # data_forward['avg_entropy'] = entropy_average

    return data_forward

"""
# back propagation
def back_prop(data):
    # 1. softmax and cross entropy
    grad_en_ak = (data['y'] - network['t_label_one_hot'].T) / batch_size

    # 2. find grad(E_n, X), grad(E_n, W2), grad(E_n, b2)
    grad_en_x2 = np.dot(network['w2'].T, grad_en_ak)
    grad_en_w2 = np.dot(grad_en_ak, data['z1'].T)
    grad_en_b2 = grad_en_ak.sum(axis=1)
    grad_en_b2 = grad_en_b2.reshape((c, 1))

    # 3. back propagate : sigmoid
    bp_sigmoid = np.dot(network['w2'].T, grad_en_ak) * (1 - f_sigmoid(data['a1'])) * f_sigmoid(data['a1'])

    grad_en_x1 = np.dot(network['w1'].T, bp_sigmoid)
    grad_en_w1 = np.dot(bp_sigmoid, data['x1'].T)
    grad_en_b1 = bp_sigmoid.sum(axis=1)
    grad_en_b1 = grad_en_b1.reshape((m, 1))

    # 5. update parameter
    network['w1'] -= eta * grad_en_w1
    network['w2'] -= eta * grad_en_w2
    network['b1'] -= eta * grad_en_b1
    network['b2'] -= eta * grad_en_b2

"""

# main
if __name__ == "__main__":
    # --- number input ---
    while True:
        print("input an integer from 0 to 9999.")
        idx = int(sys.stdin.readline(), 10)
        if 0 <= idx < 10000:
            break
        else:
            print("invalid input ;-(")

    input_img = X[idx]
    network['t_label'] = Y[idx]

    # forwarding
    forward_data = forward()
    y = np.argmax(forward_data['y'], axis=0)
    
    # -- for showing images --
    plt.imshow(input_img, cmap=cm.gray)
    print("Recognition result -> {0} \n Correct answer -> {1}".format(y, network['t_label']))
    plt.show()