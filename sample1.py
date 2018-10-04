import numpy as np
import sys
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

mndata = MNIST("/Users/lugh/Desktop/KUInfo/winter-CS3rd/le4-dip/works")

# settings
img_size = 28 * 28
img_len = 10000
class_num = 10
mid_layer_num = 100


# input_layer
def input_layer (in_x):
    return in_x.reshape(img_size)

# mid_layer


# output_layer


if __name__ == "__main__":
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0],28,28))
    Y = np.array(Y)


    # stdin
    print ("input an integer from 0 to 9999.")
    idx = int(sys.stdin.readline(), 10)

    print (X[idx])

    new_x = input_layer(X[idx])

    print ("new_x is below")
    print (new_x)
    plt.imshow(X[idx], cmap=cm.gray)
    plt.show()
    print (Y[idx])
