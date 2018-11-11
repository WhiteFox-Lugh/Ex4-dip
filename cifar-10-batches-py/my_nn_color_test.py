import numpy as np
from numpy import ndarray
import math
import sys
import matplotlib.pyplot as plt
from pylab import cm
import cPickle as cPickle
from progressbar import ProgressBar


class NNColorTest:
    """ Class of Neural Network (Test) color ver.

    Attributes:
        network: data and parameters in NN.

    """
    d = 3 * 32 * 32
    img_div = 255
    c = 10
    eta = 0.01
    m = 1000
    batch_size = 1
    per_epoch = 10000 // batch_size
    epoch = 100
    testtime = 100
    p = ProgressBar()
    f = "./test_batch"
    with open(f, 'rb') as fo:
        dict = cPickle.load(fo)
    data_x = np.array(dict['data'])
    data_x = data_x.reshape((data_x.shape[0], 3, 32, 32))
    data_y = np.array(dict['labels'])
    bp_param = {}
    # --- for Adam ---
    bp_param['adam_t'] = 0.0
    bp_param['adam_m1'] = 0.0
    bp_param['adam_m2'] = 0.0
    bp_param['adam_v1'] = 0.0
    bp_param['adam_v2'] = 0.0
    bp_param['adam_beta1'] = 0.9
    bp_param['adam_beta2'] = 0.999
    bp_param['adam_epsilon'] = 10.0 ** (-8)
    bp_param['adam_alpha'] = 0.001

    def __init__(self):
        self.network = {}
        w1_tmp = np.random.normal(0.0, math.sqrt(1.0 / self.d), self.m * self.d)
        self.network['w1'] = w1_tmp.reshape((self.m, self.d))
        w2_tmp = np.random.normal(0.0, math.sqrt(1.0 / self.m), self.c * self.m)
        self.network['w2'] = w2_tmp.reshape((self.c, self.m))
        b1_tmp = np.random.normal(0.0, math.sqrt(1.0 / self.d), self.m)
        self.network['b1'] = b1_tmp.reshape((self.m, 1))
        b2_tmp = np.random.normal(0, math.sqrt(1.0 / self.m), self.c)
        self.network['b2'] = b2_tmp.reshape((self.c, 1))


def f_sigmoid(t):
    """ Apply sigmoid function.

    Args:
        t: The input value of this function.

    Returns:
        The output of sigmoid function.

    """
    return 1.0 / (1.0 + np.exp(-t))


def relu_forward(t):
    """ Apply ReLU function.

    Args:
        t: The input value of this function.

    Returns:
        The output of ReLU function.
    """

    return np.maximum(t, 0.0)


def relu_backward(t):
    """ find gradient of ReLU function.

    Args:
        t: The input value of this function.

    Returns:
        The output of this function.

    """
    return np.where(t > 0.0, 1.0, 0.0)


def f_softmax(t):
    """ Apply softmax function.

    Args:
        t: The input value of this function.

    Returns:
        The output of this function.

    """
    alpha = np.max(t) * 1.0
    r = (np.exp(t - alpha) * 1.0) / np.sum(np.exp(t - alpha))
    return r


def input_layer(nn, x):
    """ Input layer of NN.

    Args:
        x: Array of image data [shape -> (1, d = 3 * 32 * 32)].
        nn: Extends NNTest class.

    Returns:
        Normalized and reshaped Array of MNIST image data (type: ndarray).
        The shape of array is (d, 1).

    """
    tmp = x.reshape((nn.batch_size, nn.d))
    return (tmp.T * 1.0) / nn.img_div


def affine_transformation(w, x, b):
    """ Affine transformation in hidden layer.

    Args:
        w: Weight
        x: Input data
        b: Bias

    Returns:
        The array applied affine transformation function.

    """
    return np.dot(w * 1.0, x) + b


def mid_layer_activation(t):
    """ Apply activation function in hidden layer.

    Args:
        t: input from previous affine layer.

    Returns:
        The array applied activation function.
        The shape of array is (m, 1).
    """
    return np.apply_along_axis(relu_forward, axis=0, arr=t)


def output_layer_apply(t):
    """ Apply softmax function in output layer.

    Args:
        t: input from previous affine layer.

    Returns:
        The array applied activation function (softmax function).
        The shape of array is (c, 1).

    """
    return np.apply_along_axis(f_softmax, axis=0, arr=t)


def one_hot_vector(t, c):
    """ Make one-hot vector

    Args:
        t: correct label
        c: number of class

    Returns:
        correct label (in one-hot vector expression)

    """
    return np.identity(c)[t]


def cal_cross_entropy(nn, prob, label):
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
        tmp_e = 0.0
        for k in range(nn.c):
            tmp_e += (-label[j][k] * np.log(y_p[j][k])) * 1.0
        e = np.append(e, tmp_e)
    return e


def adam(nn, bp_data):
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


def forward(nn, input_img):
    """ Forwarding

    Args:
        nn: Class NNLearn
        input_img: selected images

    Returns:
        Dictionary data including the calculation result in each layer.

    """

    data_forward = {}

    # input_layer : (1, batch_size * d) -> (d, batch_size)
    output_input_layer = input_layer(nn, input_img)

    # mid_layer : (d, batch_size) -> (m, batch_size)
    a_mid_layer = affine_transformation(nn.network['w1'], output_input_layer, nn.network['b1'])
    z_mid_layer = mid_layer_activation(a_mid_layer)

    # output_layer : (m, batch_size) -> (c = 10, batch_size)
    a_output_layer = affine_transformation(nn.network['w2'], z_mid_layer, nn.network['b2'])
    result = output_layer_apply(a_output_layer)

    # find cross entropy
    # entropy = cal_cross_entropy(nn, result, nn.network['t_label_one_hot'])
    # entropy_average = np.sum(entropy, axis=0) / nn.batch_size

    data_forward['x1'] = output_input_layer
    data_forward['a1'] = a_mid_layer
    data_forward['z1'] = z_mid_layer
    data_forward['a2'] = a_output_layer
    data_forward['y'] = result
    # data_forward['avg_entropy'] = entropy_average

    return data_forward


def main():
    nn = NNColorTest()
    # load parameter
    err_time = 0
    while True:
        try:
            if err_time >= 3:
                print("Process finished...")
                sys.exit(0)

            print("Input the filename of the parameter file.")
            print("If you have no parameter file, push the Enter key.")
            filename = str(sys.stdin.readline())
            filename = filename.replace('\n', '')
            filename = filename.replace('\r', '')
            if filename == "":
                print("Randomize parameter")
                break
            load_param = np.load(filename)

        except Exception as e:
            print("Error: {0}".format(e))
            err_time = err_time + 1

        else:
            nn.network['w1'] = load_param['w1']
            nn.network['w2'] = load_param['w2']
            nn.network['b1'] = load_param['b1']
            nn.network['b2'] = load_param['b2']
            break

    correct = 0
    incorrect = 0
    iteration_test = []
    accuracy_test = []
    nums = list(range(0, nn.data_x.size // nn.d))

    for itr in range(nn.testtime):
        # init
        input_img = np.array([], dtype='int32')
        t_label = np.array([], dtype='int32')

        # select from training data
        choice_nums = np.random.choice(nums, nn.batch_size, replace=False)

        # data input
        for i in range(nn.batch_size):
            tmp_img = nn.data_x[choice_nums[i]]
            input_img = np.append(input_img, tmp_img)
            t_label = np.append(t_label, nn.data_y[choice_nums[i]])

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

    accuracy = correct * 1.0 / (correct + incorrect)
    print("accuracy -> {0}".format(accuracy))


if __name__ == "__main__":
    main()

