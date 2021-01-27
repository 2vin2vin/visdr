#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module Name:        AirodynCV/networks
#   Submodule Name:     networks/CNN
#   File Name:          shallownet.py
#   Author:             Aviral Sharma
# ______________________________________
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        # basically to check whether using channels_first or channels_last
        input_shape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # Defining Convolution Layer, No of Kernels = 32, Kernel Size = 3x3
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=input_shape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

# _________________________________________________ END OF FILE __________________________________________________ #
