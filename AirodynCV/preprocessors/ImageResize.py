#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module Name:        AirodynCV
#   Submodule Name:     preprocessors
#   File Name:          ImageResize.py
#   Author:             Aviral Sharma
# ______________________________________

# Import necessary packages
import cv2


# Class for image resize
class ImageResize:
    """Class to resize images to fixed size for data training. Inputs the target image size, and interpolation method"""
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # Target image's width, height and interpolation method for resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """Resize image using OpenCV resize method ignoring the Aspect Ratio of the Image."""
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

# _________________________________________________ END OF FILE __________________________________________________ #
