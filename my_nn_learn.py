import numpy as np
import math
from numpy import ndarray
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from progressbar import ProgressBar

mndata = MNIST("/Users/lugh/Desktop/KUInfo/winter-CS3rd/le4-dip/works")
# mndata = MNIST("/export/home/016/a0167009/le4-dip/Ex4-dip")
p = ProgressBar()


class NNLearn:
    """ Class of Neural Network (learning).

    Attributes:
        network: data and parameters in NN.

    """
    d = 28 * 28
    img_div = 255
    c = 10
    eta = 0.01
    m = 200
    batch_size = 100
    per_epoch = 60000 // batch_size
    epoch = 30
    p = ProgressBar()
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)
    mode = {}
    # --- for Momentum SGD ---
    alpha = 0.9
    bp_param = {}
    bp_param['msgd_w1'] = np.zeros((m, d))
    bp_param['msgd_w2'] = np.zeros((c, m))
    # --- for AdaGrad ---
    bp_param['ag_h1'] = 10 ** (-8)
    bp_param['ag_h2'] = 10 ** (-8)
    bp_param['lr'] = 0.001
    # --- for RMSProp ---
    bp_param['rmsprop_h1'] = 0
    bp_param['rmsprop_h2'] = 0
    bp_param['rmsprop_rho'] = 0.9
    bp_param['rmsprop_epsilon'] = 10 ** (-8)
    # --- for AdaDelta ---
    bp_param['adadelta_h1'] = 0
    bp_param['adadelta_h2'] = 0
    bp_param['adadelta_s1'] = 0
    bp_param['adadelta_s2'] = 0
    bp_param['adadelta_rho'] = 0.95
    bp_param['adadelta_epsilon'] = 10 ** (-6)
    # --- for Adam ---
    bp_param['adam_t'] = 0
    bp_param['adam_m1'] = 0
    bp_param['adam_m2'] = 0
    bp_param['adam_v1'] = 0
    bp_param['adam_v2'] = 0
    bp_param['adam_beta1'] = 0.9
    bp_param['adam_beta2'] = 0.999
    bp_param['adam_epsilon'] = 10 ** (-8)
    bp_param['adam_alpha'] = 10 ** (-3)

    def __init__(self):
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

    def __init__(self, nn: NNLearn):
        self.dropout_num = np.random.choice(nn.batch_size, int(nn.batch_size * self.rho), replace=False)
        self.mask = np.zeros((nn.m, nn.batch_size))

    def gen_mask(self, nn: NNLearn) -> ndarray:
        tmp1 = np.identity(nn.batch_size)[self.dropout_num]
        tmp2 = np.sum(tmp1, axis=0)
        tmp3 = np.repeat(tmp2, nn.m)
        tmp4 = tmp3.reshape(nn.batch_size, nn.m)
        return 1 - tmp4.T

    def forward(self, nn: NNLearn, t: ndarray) -> ndarray:
        self.mask = self.gen_mask(nn)
        return t * self.mask

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


def relu_backward(t: ndarray) -> ndarray:
    """ find gradient of ReLU function.

    Args:
        t: The input value of this function.

    Returns:
        The output of this function.

    """
    return np.where(t > 0, 1, 0)


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


def input_layer(nn: NNLearn, x: ndarray) -> ndarray:
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
    elif mode == 1:
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


def cal_cross_entropy(nn: NNLearn, prob: ndarray, label: ndarray) -> ndarray:
    """ Calculate cross entropy

    Args:
        nn: NNLearn
        prob: output from output layer
        label: correct label (one-hot vector expression)

    Returns:
        cross entropy value of each recognition result

    """

    e = np.array([], dtype="float32")
    y_p = prob.T
    for j in range(nn.batch_size):
        tmp_e = 0
        for k in range(nn.c):
            tmp_e += (-label[j][k] * np.log(y_p[j][k]))
        e = np.append(e, tmp_e)
    return e


def sgd(nn: NNLearn, bp_data: dict):
    """ Applying Stochastic Gradient Descent (SGD).

    Args:
        nn: Class NNLearn
        bp_data: data of back propagation

    """
    nn.network['w1'] -= nn.eta * bp_data['g_en_w1']
    nn.network['w2'] -= nn.eta * bp_data['g_en_w2']
    nn.network['b1'] -= nn.eta * bp_data['g_en_b1']
    nn.network['b2'] -= nn.eta * bp_data['g_en_b2']


def momentum_sgd(nn: NNLearn, bp_data: dict):
    """ Applying Momentum Stochastic Gradient Descent.

    Args:
        nn: Class NNLearn
        bp_data: data of back propagation

    """
    nn.bp_param['msgd_w1'] = (nn.alpha * nn.bp_param['msgd_w1']) - (nn.eta * bp_data['g_en_w1'])
    nn.bp_param['msgd_w2'] = (nn.alpha * nn.bp_param['msgd_w2']) - (nn.eta * bp_data['g_en_w2'])
    nn.network['w1'] += nn.bp_param['msgd_w1']
    nn.network['w2'] += nn.bp_param['msgd_w2']
    nn.network['b1'] -= nn.eta * bp_data['g_en_b1']
    nn.network['b2'] -= nn.eta * bp_data['g_en_b2']


def adagrad(nn: NNLearn, bp_data: dict):
    """ Applying AdaGrad

    Args:
        nn: Class NNLearn
        bp_data: date of back propagation

    """
    nn.bp_param['ag_h1'] = nn.bp_param['ag_h1'] + (bp_data['g_en_w1'] * bp_data['g_en_w1'])
    nn.bp_param['ag_h2'] = nn.bp_param['ag_h2'] + (bp_data['g_en_w2'] * bp_data['g_en_w2'])
    nn.network['w1'] -= (nn.bp_param['lr'] / np.sqrt(nn.bp_param['ag_h1'])) * bp_data['g_en_w1']
    nn.network['w2'] -= (nn.bp_param['lr'] / np.sqrt(nn.bp_param['ag_h2'])) * bp_data['g_en_w2']
    nn.network['b1'] -= nn.eta * bp_data['g_en_b1']
    nn.network['b2'] -= nn.eta * bp_data['g_en_b2']


def rms_prop(nn: NNLearn, bp_data: dict):
    """ Applying RMSProp

    Args:
        nn: Class NNLearn
        bp_data: date of back propagation

    """

    rho = nn.bp_param['rmsprop_rho']
    epsilon = nn.bp_param['rmsprop_epsilon']
    nn.bp_param['rmsprop_h1'] = rho * nn.bp_param['rmsprop_h1'] +\
                                (1 - rho) * (bp_data['g_en_w1'] * bp_data['g_en_w1'])
    nn.bp_param['rmsprop_h2'] = rho * nn.bp_param['rmsprop_h2'] +\
                                (1 - rho) * (bp_data['g_en_w2'] * bp_data['g_en_w2'])
    nn.network['w1'] -= (nn.bp_param['lr'] / (np.sqrt(nn.bp_param['rmsprop_h1']) + epsilon)) * bp_data['g_en_w1']
    nn.network['w2'] -= (nn.bp_param['lr'] / (np.sqrt(nn.bp_param['rmsprop_h2']) + epsilon)) * bp_data['g_en_w2']
    nn.network['b1'] -= nn.eta * bp_data['g_en_b1']
    nn.network['b2'] -= nn.eta * bp_data['g_en_b2']


def ada_delta(nn: NNLearn, bp_data: dict):
    """ Applying AdaDelta

    Args:
        nn: Class NNLearn
        bp_data: date of back propagation

    """

    rho = nn.bp_param['adadelta_rho']
    nn.bp_param['adadelta_h1'] = rho * nn.bp_param['adadelta_h1'] +\
                           (1 - rho) * (bp_data['g_en_w1'] * bp_data['g_en_w1'])
    nn.bp_param['adadelta_h2'] = rho * nn.bp_param['adadelta_h2'] +\
                           (1 - rho) * (bp_data['g_en_w2'] * bp_data['g_en_w2'])
    nn.bp_param['adadelta_dw1'] = ((-np.sqrt(nn.bp_param['adadelta_s1'] + nn.bp_param['adadelta_epsilon']))
                                   / np.sqrt(nn.bp_param['adadelta_h1'] + nn.bp_param['adadelta_epsilon']))\
                                  * bp_data['g_en_w1']
    nn.bp_param['adadelta_dw2'] = ((-np.sqrt(nn.bp_param['adadelta_s2'] + nn.bp_param['adadelta_epsilon']))
                                   / np.sqrt(nn.bp_param['adadelta_h2'] + nn.bp_param['adadelta_epsilon']))\
                                  * bp_data['g_en_w2']
    nn.bp_param['adadelta_s1'] = rho * nn.bp_param['adadelta_s1'] +\
                           (1 - rho) * (nn.bp_param['adadelta_dw1'] * nn.bp_param['adadelta_dw1'])
    nn.bp_param['adadelta_s2'] = rho * nn.bp_param['adadelta_s2'] +\
                           (1 - rho) * (nn.bp_param['adadelta_dw2'] * nn.bp_param['adadelta_dw2'])
    nn.network['w1'] += nn.bp_param['adadelta_dw1']
    nn.network['w2'] += nn.bp_param['adadelta_dw2']
    nn.network['b1'] -= nn.eta * bp_data['g_en_b1']
    nn.network['b2'] -= nn.eta * bp_data['g_en_b2']


def adam(nn: NNLearn, bp_data: dict):
    """ Applying Adam

    Args:
        nn: Class NNLearn
        bp_data: date of back propagation

    """
    nn.bp_param['adam_t'] = nn.bp_param['adam_t'] + 1
    beta1 = nn.bp_param['adam_beta1']
    beta2 = nn.bp_param['adam_beta2']
    nn.bp_param['adam_m1'] = beta1 * nn.bp_param['adam_m1'] + (1 - beta1) * bp_data['g_en_w1']
    nn.bp_param['adam_m2'] = beta1 * nn.bp_param['adam_m2'] + (1 - beta1) * bp_data['g_en_w2']
    nn.bp_param['adam_v1'] = beta2 * nn.bp_param['adam_v1'] + (1 - beta2) * bp_data['g_en_w1'] * bp_data['g_en_w1']
    nn.bp_param['adam_v2'] = beta2 * nn.bp_param['adam_v2'] + (1 - beta2) * bp_data['g_en_w2'] * bp_data['g_en_w2']
    m_hat1 = nn.bp_param['adam_m1'] / (1 - beta1 ** nn.bp_param['adam_t'])
    m_hat2 = nn.bp_param['adam_m2'] / (1 - beta1 ** nn.bp_param['adam_t'])
    v_hat1 = nn.bp_param['adam_v1'] / (1 - beta2 ** nn.bp_param['adam_t'])
    v_hat2 = nn.bp_param['adam_v2'] / (1 - beta2 ** nn.bp_param['adam_t'])
    nn.network['w1'] -= (nn.bp_param['adam_alpha'] * m_hat1) / (np.sqrt(v_hat1) + nn.bp_param['adam_epsilon'])
    nn.network['w2'] -= (nn.bp_param['adam_alpha'] * m_hat2) / (np.sqrt(v_hat2) + nn.bp_param['adam_epsilon'])
    nn.network['b1'] -= nn.eta * bp_data['g_en_b1']
    nn.network['b2'] -= nn.eta * bp_data['g_en_b2']


def forward(nn: NNLearn, input_img: ndarray):
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

    if nn.mode['mid_act_fun'] != 2:
        z_mid_layer = mid_layer_activation(nn.mode['mid_act_fun'], a_mid_layer)

    else:
        data_forward['dropout_class'] = Dropout(nn)
        z_mid_layer = data_forward['dropout_class'].forward(nn, a_mid_layer)

    # output_layer : (m, batch_size) -> (c = 10, batch_size)
    a_output_layer = affine_transformation(nn.network['w2'], z_mid_layer, nn.network['b2'])
    result = output_layer_apply(a_output_layer)

    # find cross entropy
    entropy = cal_cross_entropy(nn, result, nn.network['t_label_one_hot'])
    entropy_average = np.sum(entropy, axis=0) / nn.batch_size

    data_forward['x1'] = output_input_layer
    data_forward['a1'] = a_mid_layer
    data_forward['z1'] = z_mid_layer
    data_forward['a2'] = a_output_layer
    data_forward['y'] = result
    data_forward['avg_entropy'] = entropy_average

    return data_forward


def back_prop(nn: NNLearn, data: dict):
    """ Back propagation

    Args:
        nn: NNLearn
        data: calculating result in forwarding

    """
    bp_data = {}

    # 1. softmax and cross entropy
    bp_data['g_en_ak'] = (data['y'] - nn.network['t_label_one_hot'].T) / nn.batch_size

    # 2. find grad(E_n, X), grad(E_n, W2), grad(E_n, b2)
    bp_data['g_en_x2'] = np.dot(nn.network['w2'].T, bp_data['g_en_ak'])
    bp_data['g_en_w2'] = np.dot(bp_data['g_en_ak'], data['z1'].T)
    grad_en_b2 = bp_data['g_en_ak'].sum(axis=1)
    bp_data['g_en_b2'] = grad_en_b2.reshape((nn.c, 1))

    # 3. back propagate : activate layer
    if nn.mode['mid_act_fun'] == 0:
        # sigmoid function ver.
        bp_data['g_activate_mid'] = np.dot(nn.network['w2'].T, bp_data['g_en_ak']) *\
                      (1 - f_sigmoid(data['a1'])) * f_sigmoid(data['a1'])
    elif nn.mode['mid_act_fun'] == 1:
        # relu function ver.
        bp_data['g_activate_mid'] = np.dot(nn.network['w2'].T, bp_data['g_en_ak']) * relu_backward(data['a1'])

    elif nn.mode['mid_act_fun'] == 2:
        # dropout ver.
        bp_data['g_activate_mid'] = np.dot(nn.network['w2'].T, bp_data['g_en_ak']) \
                                    * data['dropout_class'].backward()

    # 4. find grad(E_n, X), grad(E_n, W1), grad(E_n, b2)
    bp_data['g_en_x1'] = np.dot(nn.network['w1'].T, bp_data['g_activate_mid'])
    bp_data['g_en_w1'] = np.dot(bp_data['g_activate_mid'], data['x1'].T)
    grad_en_b1 = bp_data['g_activate_mid'].sum(axis=1)
    bp_data['g_en_b1'] = grad_en_b1.reshape((nn.m, 1))

    # 5. update parameter
    if nn.mode['param_update'] == 0:
        sgd(nn, bp_data)
    elif nn.mode['param_update'] == 1:
        momentum_sgd(nn, bp_data)
    elif nn.mode['param_update'] == 2:
        adagrad(nn, bp_data)
    elif nn.mode['param_update'] == 3:
        rms_prop(nn, bp_data)
    elif nn.mode['param_update'] == 4:
        ada_delta(nn, bp_data)
    elif nn.mode['param_update'] == 5:
        adam(nn, bp_data)


def main():
    """
    This is the main function.
    """

    nn = NNLearn()
    nums = list(range(0, nn.X.size // nn.d))

    loss = np.array([])
    iteration = np.array([], dtype='int32')

    # mode settings
    err_time = 0
    while True:
        if err_time >= 3:
            print("プログラムを終了します...")
            sys.exit(0)

        try:
            print("中間層の活性化関数(またはDropout)を選択してください")
            print("0 -> sigmoid, 1 -> ReLU, 2 -> Dropout")
            tmp = int(sys.stdin.readline(), 10)
            if 0 <= tmp <= 2:
                nn.mode['mid_act_fun'] = tmp
            else:
                raise Exception("Error: 無効な入力です")

            print("最適化手法を選択してください")
            print("0 -> SGD, 1 -> Momentum SGD, 2 -> AdaGrad,")
            print("3 -> RMSProp, 4 -> AdaDelta, 5 -> Adam")
            tmp = int(sys.stdin.readline(), 10)
            if 0 <= tmp <= 5:
                nn.mode['param_update'] = tmp
            else:
                raise Exception("Error: 無効な入力です")

            break

        except Exception as e:
            err_time = err_time + 1
            print(e)

    for itr in p(range(nn.per_epoch * nn.epoch)):
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

        # print cross entropy
        # print("average cross entropy -> {0}".format(forward_data['avg_entropy']))

        # back propagation
        back_prop(nn, forward_data)

        iteration = np.append(iteration, itr + 1)
        loss = np.append(loss, forward_data['avg_entropy'])

        if itr % nn.per_epoch == nn.per_epoch - 1:
            plt.plot(iteration, loss)
            plt.title("entropy")
            plt.xlabel("itr")
            plt.ylabel("entropy")
            plt.show()

    # save parameters
    print("パラメータを保存するファイルの名前を ***.npz の形式で入力してください")
    filename = str(sys.stdin.readline())
    filename = filename.replace('\n', '')
    filename = filename.replace('\r', '')
    np.savez(filename, w1=nn.network['w1'], w2=nn.network['w2'], b1=nn.network['b1'],
             b2=nn.network['b2'], loss=loss)


if __name__ == "__main__":
    main()
