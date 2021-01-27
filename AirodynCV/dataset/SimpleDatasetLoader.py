#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module Name:        AirodynCV
#   Submodule Name:     dataset
#   File Name:          SimpleDatasetLoader.py
#   Author:             Aviral Sharma
# ______________________________________

# import necessary packages
import numpy as np
import cv2
import os


# Defining class to load dataset
class SimpleDatasetLoader:
    """MAKE SURE THAT ALL THE IMAGES IN THE DATASET CAN FIT IN RAM"""
    def __init__(self, preprocessors=None):
        # initialising preprocessors from preprocessors submodule
        self.preprocessors = preprocessors

        # if preprocessors = None, initialize as empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize list of images and their label categories
        """Load all images from given directory, apply passed pre-processors sequentially, one at a time, and store
         the images and their corresponding labels as Numpy Arrays.
         Displays progressbar for image processing."""
        data = []
        labels = []

        # looping over input images to add images to data list and their labels to labels list
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]        # The images are stored as /dataset_name/class/image.jpg

            # check to see if preprocessors are None
            if self.preprocessors is not None:
                # loop over preprocessors and apply one at a time
                for p in self.preprocessors:
                    image = p.preprocess(image)

        # Adding preprocessed image to data and its label to label
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] Processed {}/{}".format(i+1, len(imagePaths)))

        return np.array(data), np.array(labels)

# _________________________________________________ END OF FILE __________________________________________________ #
