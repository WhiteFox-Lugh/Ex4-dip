import numpy as np
import math
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

mndata = MNIST("/Users/lugh/Desktop/KUInfo/winter-CS3rd/le4-dip/works")

# settings
d = 28 * 28
img_div = 255
img_len = 10000
c = 10
# mid layer num
m = 30
batch_size = 100


# init weight and bias
np.random.seed(3304)
w1 = np.random.normal(0, math.sqrt(1 / d), m * d)
w1 = w1.reshape((m, d))
w2 = np.random.normal(0, math.sqrt(1 / m), c * m)
w2 = w2.reshape((c, m))
b1 = np.random.normal(0, math.sqrt(1 / d), m)
b2 = np.random.normal(0, math.sqrt(1 / m), c)


# input_layer : (1, batch_size * d) -> (d = 784, batch_size = 100)
def input_layer(in_x):
    tmp = in_x.reshape(batch_size, d)
    return tmp.T / img_div


# mid_layer : (d = 784, batch_size = 100) -> (m = 30, batch_size = 100)
def mid_layer(in_x):
    result = np.apply_along_axis(lambda l: f_sigmoid(np.dot(w1, l) + b1), 0, [in_x[i] for i in range(in_x.shape[0])])
    return result


# output_layer : (m = 30, batch_size = 100) -> (c = 10, batch_size = 100)
def output_layer(in_x):
    result = np.apply_along_axis(lambda l: f_softmax(np.dot(w2, l) + b2), 0, [in_x[i] for i in range(in_x.shape[0])])
    return result


# sigmoid function
def f_sigmoid(t):
    return 1 / (1 + np.exp((-1) * t))


# softmax function
def f_softmax(t):
    alpha = np.max(t)
    result = np.exp(t - alpha) / np.sum(np.exp(t - alpha))
    return result


# one-hot vector
def one_hot_vector(t):
    return np.identity(10)[t]


# calculate cross entropy
def cal_cross_entropy(y, label):
    e = 0
    for i in range(batch_size):
        for j in range(c):
            e += (-label[i][j] * np.log(y[j]))

    return e


# print result
def show_result(t):
    print("******* result *******")
    entropy = cal_cross_entropy(result, t_label_one_hot)
    entropy_average = np.sum(entropy) / batch_size
    print("Cross entropy av: {0}".format(entropy_average))
    for i in range(batch_size):
        print("Img:{0} -> Class:{1}".format(i, np.argmax(t.T[i])))


# main
if __name__ == "__main__":
    # loading data
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)

    # random choice
    nums = list(range(0, int((X.size / d))))
    choice_nums = np.random.choice(nums, batch_size, replace=False)

    input_img = []
    t_label = []
    for i in range(batch_size):
        input_img = np.hstack((input_img, X[choice_nums[i]].reshape(d)))
        t_label = t_label + [ Y[choice_nums[i]] ]

    # make one-hot vector
    t_label_one_hot = one_hot_vector(t_label)

    # 3-layer NN main
    output_input_layer = input_layer(input_img)
    output_mid_layer = mid_layer(output_input_layer)
    result = output_layer(output_mid_layer)
    show_result(result)

    #plt.imshow(X[idx], cmap=cm.gray)
    #plt.show()
    # print(Y[idx])

