#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module:             AirodynCV
#   Sub Module:         networks/CNN
#   File Name:          DenseHead.py
#   Author:             Aviral Sharma
# ______________________________________

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class DenseHead:
    @staticmethod
    def build(baseModel, classes, D):   # D = number of nodes in fully connected layer
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel
