import numpy as np
import keras
import time
from keras import datasets, models, layers
import tensorflow as tf
from keras.backend import set_session, tensorflow_backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def timestamp():
    """ make timestamp :-) """
    return time.strftime("%Y-%m%d-%H_%M_%S")

def main():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list="0"))
    set_session(tf.Session(config=config))

    img_rows, img_cols = 28, 28
    num_classes = 10

    (X, Y), (Xtest, Ytest) = keras.datasets.mnist.load_data()

    # X -> (ID, 28, 28, 1)
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)

    # normalize
    X = X.astype('float32') / 255.0
    Xtest = Xtest.astype('float32') / 255.0

    input_shape = (img_rows, img_cols, 1)

    # label -> one-hot vector
    Y = keras.utils.to_categorical(Y, num_classes)
    Ytest1 = keras.utils.to_categorical(Ytest, num_classes)

    # define model
    model = models.Sequential()

    # 3x3 Convolution, output -> 32 channels, activation function -> ReLU
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))

    # 3x3 Convolution, output -> 64 channels, activation function -> ReLU
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    # 2x2 max pooling
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # flatten
    model.add(layers.Flatten())

    # fully connected layer, output node -> 128, activation function -> ReLU
    model.add(layers.Dense(128, activation='relu'))

    # fully connected layer, output node -> num_classes, activation function -> softmax
    model.add(layers.Dense(num_classes, activation='softmax'))

    # show summary
    model.summary()

    # compile
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['acc'])

    # learning
    epochs = 10
    batch_size = 128
    result = model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(Xtest, Ytest1),
                   callbacks=[keras.callbacks.TensorBoard(log_dir='./logs/res/', histogram_freq=1, batch_size=128)])

    history = result.history

    # save
    model.save("model_" + timestamp() + ".h5")

    # load model
    # model = models.load_model("my_model.h5")

    # load learning history
    # with open("my_model_history.dump", "rb") as f:
    # history = pickle.load(f)

    # Xtest pred
    pred = model.predict_classes(Xtest)

    print (confusion_matrix(Ytest, pred, labels=np.arange(10)))

    # print accuracy
    print (accuracy_score(Ytest, pred))

    # print loss function and graph
    fig = plt.figure()
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig("loss_history_" + timestamp() + ".png")

    fig = plt.figure()
    plt.plot(history['acc'], label='acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.savefig("loss_acc_" + timestamp() + ".png")


if __name__ == "__main__":
    main()
