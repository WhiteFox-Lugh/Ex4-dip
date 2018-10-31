import numpy as np
from numpy import ndarray
import math
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from progressbar import ProgressBar

mndata = MNIST("./")
p = ProgressBar()


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
    per_epoch = 10000 // batch_size
    filename = ""
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


def f_bn_test(nn: NNTest, t: ndarray) -> ndarray:
    def trans_y(t):
        return (nn.network['gamma'] / np.sqrt(nn.network['exp_var'] + nn.network['eps'])) * t +\
               (nn.network['beta'] - (nn.network['gamma'] * nn.network['exp_avg']) /
                np.sqrt(nn.network['exp_var'] + nn.network['eps']))

    r = np.apply_along_axis(trans_y, axis=0, arr=t)
    return r


def input_layer(nn: NNTest, x: ndarray) -> ndarray:
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


def mid_layer_activation(mode: int, t: ndarray) -> ndarray:
    """ Apply activation function in hidden layer.

    Args:
        mode: decide which activation function is used.
        t: input from previous affine layer.

    Returns:
        The array applied activation function.
        The shape of array is (m, 1).
    """

    if mode == 0:
        return np.apply_along_axis(f_sigmoid, axis=0, arr=t)
    else:
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


def one_hot_vector(t: ndarray, c: int) -> ndarray:
    """ Make one-hot vector

    Args:
        t: correct label
        c: number of class

    Returns:
        correct label (in one-hot vector expression)

    """
    return np.identity(c)[t]


def forward(nn: NNTest, input_img: ndarray):
    """ Forwarding

    Args:
        nn: Class NNLearn
        input_img: selected images

    Returns:
        Dictionary data including the calculation result in each layer.

    """

    data_forward = {}

    # input_layer : (1, batch_size * d) -> (d = 784, batch_size)
    output_input_layer = input_layer(nn, input_img)

    # mid_layer : (d = 784, batch_size) -> (m, batch_size)
    a_mid_layer = affine_transformation(nn.network['w1'], output_input_layer, nn.network['b1'])

    # apply Batch normalization
    if nn.network['mode'] == 3:
        a_mid_normalize = f_bn_test(nn, a_mid_layer)
        z_mid_layer = mid_layer_activation(nn.network['mode'], a_mid_normalize)

    # apply Dropout
    elif nn.network['mode'] == 2:
        z_mid_layer = (1 - nn.network['rho']) * a_mid_layer

    # apply neither Batch normalization nor Dropout
    else:
        z_mid_layer = mid_layer_activation(nn.network['mode'], a_mid_layer)

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

    correct = 0
    incorrect = 0
    iteration_test = []
    accuracy_test = []
    testtime = 100
    # load parameter
    while True:
        nn = NNTest()
        nums = list(range(0, nn.X.size // nn.d))
        while True:
            try:
                print("パラメータを保存してあるファイル名を入力して下さい.")
                print("読み込まない場合は何も入力せずに Enter を押してください")
                nn.filename = str(sys.stdin.readline())
                nn.filename = nn.filename.replace("\n", "")
                nn.filename = nn.filename.replace("\r", "")
                legend_name = nn.filename.replace("param_", "")
                legend_name = legend_name.replace(".npz", "")

                if nn.filename == "":
                    print("パラメータをランダムに初期化してテストを行います")
                    mode = 1
                    break

                load_param = np.load(nn.filename)
                mode = 0
                nn.batch_size = 100

                print("用いる活性化関数を選んでください.")
                print("0 -> sigmoid, 1 -> ReLU, 2 -> Dropout")
                print("3 -> ReLU + Batch Normalization")
                nn.network['mode'] = int(sys.stdin.readline(), 10)

                if not (0 <= nn.network['mode'] <= 3):
                    raise Exception("不正な入力です")

                elif nn.network['mode'] == 2:
                    nn.network['rho'] = load_param['rho']

                elif nn.network['mode'] == 3:
                    nn.network['exp_avg'] = load_param['exp_avg']
                    nn.network['exp_var'] = load_param['exp_var']
                    nn.network['beta'] = load_param['beta']
                    nn.network['gamma'] = load_param['gamma']
                    nn.network['eps'] = load_param['eps']

                print("テスト回数を入力してください(1~1000).")
                print("バッチサイズは {0}, テストデータ数は {1} です.".format(nn.batch_size, nn.X.size // nn.d))
                testtime = int(sys.stdin.readline(), 10)

                if not (1 <= testtime <= 1000):
                    raise Exception("不正な入力です")

            except Exception as e:
                print("エラー: {0}".format(e))
                nn.batch_size = 1

            else:
                nn.network['w1'] = load_param['w1']
                nn.network['w2'] = load_param['w2']
                nn.network['b1'] = load_param['b1']
                nn.network['b2'] = load_param['b2']
                itr_plot_x = load_param['loss'].size
                iteration = np.arange(0, itr_plot_x, 1)
                loss = load_param['loss']
                plt.plot(iteration, loss, label=legend_name, lw=0.5)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
                plt.title("cross entropy error")
                plt.grid(True)
                plt.xlabel("itr")
                plt.ylabel("error avg")
                plt.show()
                break

        # --- Exercise 1 ver ---
        while mode == 1:
            try:
                print("0以上9999以下の整数を1つ入力してください.")
                idx = int(sys.stdin.readline(), 10)

                if 0 <= idx < 10000:
                    # forwarding
                    nn.network['mode'] = 0
                    forward_data = forward(nn, nn.X[idx])
                    y = np.argmax(forward_data['y'], axis=0)
                    break

                else:
                    print("Error: 0以上9999以下の整数ではありません")

            except Exception as e:
                print(e)

        # --- Exercise 1 ver end ---

        # -- for showing images --
        if mode == 1:
            plt.imshow(nn.X[idx], cmap=cm.gray)
            print("Recognition result -> {0} \n Correct answer -> {1}".format(y, nn.Y[idx]))
            plt.show()

        else:
            for itr in p(range(testtime)):
                # init
                input_img = np.array([], dtype='int32')
                t_label = np.array([], dtype='int32')

                # select from training data
                choice_nums = np.random.choice(nums, nn.batch_size, replace=False)

                # data input
                for i in range(nn.batch_size):
                    tmp_img = nn.X[choice_nums[i]]
                    input_img = np.append(input_img, tmp_img)
                    t_label = np.append(t_label, nn.Y[choice_nums[i]])

                nn.network['t_label'] = t_label
                nn.network['t_label_one_hot'] = one_hot_vector(t_label, nn.c)

                # forwarding
                forward_data = forward(nn, input_img)
                res = np.argmax(forward_data['y'], axis=0)
                diff = res - t_label
                correct += np.sum(diff == 0)
                incorrect += np.sum(diff != 0)
                accuracy = correct / (correct + incorrect)
                iteration_test = np.append(iteration_test, itr + 1)
                accuracy_test = np.append(accuracy_test, accuracy)

            # --- plot accuracy ---
            plt.plot(iteration_test, accuracy_test)
            plt.title("accuracy for testdata")
            plt.grid(True)
            plt.xlabel("itr")
            plt.ylabel("accuracy")
            plt.show()
            print("accuracy -> {0}".format(accuracy))

        try:
            print("続けてテストする場合は1を入力してください. 終了する場合は0を入れてください")
            plot_continue = int(sys.stdin.readline(), 10)

            if plot_continue == 0:
                print("終了します")
                sys.exit(0)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
