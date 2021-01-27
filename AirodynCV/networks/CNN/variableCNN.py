#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module Name:        AirodynCV/networks
#   Submodule Name:     networks/CNN
#   File Name:          variableCNN.py
#   Author:             Aviral Sharma
# ______________________________________
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend

# INPUT => [[CONV => RELU =>BN]*N => POOL]*M => [FC => RELU => BN]*K => DO => FC => SOFTMAX
# 0 <= N <= 3
# M >= 0
# 0 <= K <= 2


class VariableCNN:
    """This Class builds CNN based on given parameters according to the network architecture.

     There is an option to specify different number of filters and filter size for each Conv Layer

     The network architecture is defined as:
     INPUT => [[CONV(k_number, (k_size, k_size)) => ACT =>BN]*N => POOL => DO(0.25)]*M =>
     [FC(W_number) => RELU => BN]*P => DO(0.5) => FC => SOFTMAX

     where CONV => Convolution Layer, ACT => Activation Layer, BN => Batch Normalization Layer,
     FC => Fully Connected Dense Layer, DO => Dropout Layer.

     Methods:

     build(self, width, height, depth, classes, k_number=[32], k_size=[(3, 3)], activation="relu", w_number=[512])

     width      =       width of the image

     height     =       height of the image

     depth      =       number of channels in the image

     classes    =       list of classes in the dataset

     k_number   =       list of numbers of kernels in each conv layer (default=[32])

     k_size     =       list of size of kernels in each conv layer (default=(3, 3))

     activation =       activation function type (default="relu")

     w_number   =       weights to learn in each dense layer (default=[512])"""
    def __init__(self, N=1, M=1, P=0):
        self.N = N
        self.M = M
        self.P = P
        self.chan_dim = -1

    def build(self, width, height, depth, classes, k_number=[32], k_size=[(3, 3)], activation="relu", w_number=[512]):
        model = Sequential()
        activ = activation
        input_shape = (height, width, depth)

        if backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            self.chan_dim = 1

        if len(range(self.N)) is not len(k_size):
            for r in range(len(range(self.N)) - len(k_size)):
                k_size.append(k_size[-1])

        # Adding First set of layers manually since the first convolution layer requires input_shape argument
        model.add(Conv2D(k_number[0], k_size[0], padding="same", input_shape=input_shape))
        model.add(Activation(activ))
        model.add(BatchNormalization(axis=self.chan_dim))

        # 2nd Convolution Layer onwards, looping over N, number of kernels and the kernel size; if N>1
        if self.N > 1:
            for n in range(1, self.N):
                model.add(Conv2D(k_number[0], k_size[0], padding="same"))
                model.add(Activation(activ))
                model.add(BatchNormalization(axis=self.chan_dim))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.3))

        # 2nd Set of [CONV => RELU =>BN]*N => POOL layers if M>1
        if self.M > 1:
            for m, kn, ks in zip(range(1, self.M), k_number[1:], k_size[1:]):
                for n in range(1, self.N):
                    model.add(Conv2D(kn, ks, padding="same"))
                    model.add(Activation(activ))
                    model.add(BatchNormalization(axis=self.chan_dim))
                model.add(MaxPool2D(pool_size=(2, 2)))
                model.add(Dropout(0.3))
        model.add(Flatten())

        if self.P > 0:
            if len(range(self.P)) is not len(w_number):
                for r in range(len(range(self.P)) - len(w_number)):
                    w_number.append(w_number[-1])
            for p, w in zip(range(self.P), w_number):
                model.add(Dense(w))
                model.add(Activation(activ))
                model.add(BatchNormalization(axis=self.chan_dim))
            model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
