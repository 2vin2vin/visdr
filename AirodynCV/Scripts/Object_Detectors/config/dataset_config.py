#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   File Name:          dataset_config.py
#   Author:             Aviral Sharma
# ______________________________________

import os

# Base Path for the dataset
BASE_PATH = "/home/sohith/sohith/DATASETS/LISA"

# Path to annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "train_labels.csv"])

# paths to output training and testing record files, class label file
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, "records/classes.pbtxt"])

TEST_SIZE = 0.25

CLASSES = {"Pedestrian":1,"Car":2,"Truck":3,"Van":4,"Bus":5,"Bicycle":6,"People":7,"Motor":8"Awning-tricycle":9,"Ignore":10,"Others":11,"People":12,"Tricycle":13}

