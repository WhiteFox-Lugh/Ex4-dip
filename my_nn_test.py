import numpy as np
from numpy import ndarray
import math
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

mndata = MNIST("/Users/lugh/Desktop/KUInfo/winter-CS3rd/le4-dip/works")
#mndata = MNIST("/export/home/016/a0167009/le4-dip/Ex4-dip")


class NNTest:
    """ Class of Neural Network (testing).

    Attributes:
        network: data and parameters in NN.

    """
    d = 28 * 28
    img_div = 255
    c = 10
    m = 200
    batch_size = 1
    X, Y = mndata.load_testing()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)

    def __init__(self):
        np.random.seed(3304)
        # for Exercise 1
        self.network = {}
        w1_tmp = np.random.normal(0, math.sqrt(1 / self.d), self.m * self.d)
        self.network['w1'] = w1_tmp.reshape((self.m, self.d))
        w2_tmp = np.random.normal(0, math.sqrt(1 / self.m), self.c * self.m)
        self.network['w2'] = w2_tmp.reshape((self.c, self.m))
        b1_tmp = np.random.normal(0, math.sqrt(1 / self.d), self.m)
        self.network['b1'] = b1_tmp.reshape((self.m, 1))
        b2_tmp = np.random.normal(0, math.sqrt(1 / self.m), self.c)
        self.network['b2'] = b2_tmp.reshape((self.c, 1))


class Dropout:
    rho = 0.2

    def __init__(self, nn: NNTest):
        self.dropout_num = np.random.choice(nn.batch_size, int(nn.batch_size * self.rho), replace=False)
        self.mask = np.zeros((nn.m, nn.batch_size))

    def gen_mask(self, nn: NNTest) -> ndarray:
        tmp1 = np.identity(nn.batch_size)[self.dropout_num]
        tmp2 = np.sum(tmp1, axis=0)
        tmp3 = np.repeat(tmp2, nn.m)
        tmp4 = tmp3.reshape(nn.batch_size, nn.m)
        return 1 - tmp4.T

    def forward(self, nn: NNTest, t: ndarray) -> ndarray:
        self.mask = self.gen_mask(nn)
        return (1 - self.rho) * t * self.mask

    def backward(self) -> ndarray:
        return self.mask


def f_sigmoid(t: ndarray) -> ndarray:
    """ Apply sigmoid function.

    Args:
        t: The input value of this function.

    Returns:
        The output of sigmoid function.

    """
    return 1.0 / (1.0 + np.exp(-t))


def relu_forward(t: ndarray) -> ndarray:
    """ Apply ReLU function.

    Args:
        t: The input value of this function.

    Returns:
        The output of ReLU function.
    """

    return np.maximum(t, 0)


def f_softmax(t: ndarray) -> ndarray:
    """ Apply softmax function.

    Args:
        t: The input value of this function.

    Returns:
        The output of this function.

    """
    alpha = np.max(t)
    r = np.exp(t - alpha) / np.sum(np.exp(t - alpha))
    return r


def input_layer(x: ndarray, nn: NNTest) -> ndarray:
    """ Input layer of NN.

    Args:
        x: Array of MNIST image data [shape -> (1, d = 784)].
        nn: Extends NNTest class.

    Returns:
        Normalized and reshaped Array of MNIST image data (type: ndarray).
        The shape of array is (d = 784, 1).

    """
    tmp = x.reshape(nn.batch_size, nn.d)
    return tmp.T / nn.img_div


def affine_transformation(w: ndarray, x: ndarray, b: ndarray) -> ndarray:
    """ Affine transformation in hidden layer.

    Args:
        w: Weight
        x: Input data
        b: Bias

    Returns:
        The array applied affine transformation function.

    """
    return np.dot(w, x) + b


def mid_layer_activation(t: ndarray) -> ndarray:
    """ Apply activation function in hidden layer.

    Args:
        t: input from previous affine layer.

    Returns:
        The array applied activation function.
        The shape of array is (m, 1).
    """
    # return np.apply_along_axis(f_sigmoid, axis=0, arr=t)
    return np.apply_along_axis(relu_forward, axis=0, arr=t)


def output_layer_apply(t: ndarray) -> ndarray:
    """ Apply softmax function in output layer.

    Args:
        t: input from previous affine layer.

    Returns:
        The array applied activation function (softmax function).
        The shape of array is (c, 1).

    """
    return np.apply_along_axis(f_softmax, axis=0, arr=t)


def forward(nn: NNTest, idx: int):
    """ Forwarding

    Args:
        nn: Class NNTest
        idx: input from standard input

    Returns:
        Dictionary data including the calculation result in each layer.

    """
    data_forward = {}

    # input_layer : (1, batch_size * d) -> (d = 784, batch_size)
    output_input_layer = input_layer(nn.X[idx], nn)
    # mid_layer : (d = 784, batch_size) -> (m, batch_size)
    a_mid_layer = affine_transformation(nn.network['w1'], output_input_layer, nn.network['b1'])
    z_mid_layer = mid_layer_activation(a_mid_layer)
    # data_forward['dropout_class'] = Dropout(nn)
    # z_mid_layer = data_forward['dropout_class'].forward(nn, a_mid_layer)
    # output_layer : (m, batch_size) -> (c = 10, batch_size)
    a_output_layer = affine_transformation(nn.network['w2'], z_mid_layer, nn.network['b2'])
    result = output_layer_apply(a_output_layer)

    data_forward['x1'] = output_input_layer
    data_forward['a1'] = a_mid_layer
    data_forward['z1'] = z_mid_layer
    data_forward['a2'] = a_output_layer
    data_forward['y'] = result

    return data_forward


def main():
    """
    This is the main function.
    """

    nn = NNTest()
    err_time = 0
    while True:
        if err_time >= 3:
            print("プログラムを終了します...")
            sys.exit(0)

        try:
            print("0以上9999以下の整数を1つ入力してください.")
            idx = int(sys.stdin.readline(), 10)

            if 0 <= idx < 10000:
                break
            else:
                err_time = err_time + 1
                print("Error: 0以上9999以下の整数ではありません")

        except Exception as e:
            err_time = err_time + 1
            print(e)

    # load parameter
    err_time = 0
    while True:
        try:
            if err_time >= 3:
                print("プログラムを終了します...")
                sys.exit(0)

            print("パラメータを保存してあるファイル名を入力して下さい.")
            print("読み込まない場合は何も入力せずに Enter を押してください")
            filename = str(sys.stdin.readline())
            filename = filename.replace('\n', '')
            filename = filename.replace('\r', '')
            if filename == "":
                print("パラメータをランダムに初期化してテストを行います")
                break
            load_param = np.load(filename)

        except Exception as e:
            print("エラー: {0}".format(e))
            err_time = err_time + 1

        else:
            nn.network['w1'] = load_param['w1']
            nn.network['w2'] = load_param['w2']
            nn.network['b1'] = load_param['b1']
            nn.network['b2'] = load_param['b2']
            break

    # forwarding
    forward_data = forward(nn, idx)
    y = np.argmax(forward_data['y'], axis=0)

    # -- for showing images --
    plt.imshow(nn.X[idx], cmap=cm.gray)
    print("Recognition result -> {0} \n Correct answer -> {1}".format(y, nn.Y[idx]))
    plt.show()

    #"""
    correct = 0
    incorrect = 0
    for idx in range(10000):
        forward_data = forward(nn, idx)
        y = np.argmax(forward_data['y'], axis=0)
        if int(nn.Y[idx]) == int(y):
            correct = correct + 1
        else :
            incorrect = incorrect + 1
            # print("{0}: Recognition result -> {1} \n Correct answer -> {2}".format(idx, y, nn.Y[idx]))

    accuracy = correct / (correct + incorrect)
    print("accuracy -> {0}".format(accuracy))

    # -- for showing images --
    plt.imshow(nn.X[idx], cmap=cm.gray)
    #print("Recognition result -> {0} \n Correct answer -> {1}".format(y, nn.Y[idx]))
    plt.show()
    #"""


if __name__ == "__main__":
    main()
