#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   File Name:          dataset_config.py
#   Author:             Aviral Sharma
# ______________________________________

import os

# Base Path for the dataset
BASE_PATH = "./"

# Path to annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "train_labels.csv"])

# paths to output training and testing record files, class label file
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, "records/classes.pbtxt"])

TEST_SIZE = 0.25

CLASSES = {"Pedestrian":1,"2-wheeler":2,"4-wheeler":3,"Truck":4,"Bus":5,"Other":6}

