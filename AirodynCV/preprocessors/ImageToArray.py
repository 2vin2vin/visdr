#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module Name:        AirodynCV
#   Submodule Name:     preprocessors
#   File Name:          ImageToArray.py
#   Author:             Aviral Sharma
# ______________________________________

from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArray:
    """Keras library function img_to_array properly orders the channels in the image based on image_data_format\n
    setting in the keras.json file at ~/.keras/keras.json directory according to the backend used by Keras"""
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        """The Keras utility function that correctly rearranges the dimensions of the image"""
        return img_to_array(image, data_format=self.dataFormat)

# _________________________________________________ END OF FILE __________________________________________________ #
