#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   Module Name:        AirodynCV
#   Submodule Name:     input_output
#   File Name:          hdf5DatasetWriter.py
#   Author:             Aviral Sharma
# ______________________________________

# Import necessary packages
import h5py
import os


class HDF5DatasetWriter:

    def __init__(self, dims, outputPath, dataKey="images", buffSize=1000):
        """HDF5DatasetWriter constructor accepts 4 parameters:

            Method: __init__
                dims => dimension or shape of data;
                example1: for 60,000 RGB images of size 32x32, dims = (60000, 32, 32, 3)
                example2: for VGG16 feature extraction, final output of POOL layer = 512, 7x7 = 25088; dims = (N, 25088)
                where N = number of images in dataset

                outputPath => path where HDF5 file is to be stored on disk

                dataKey => name of dataset; default='images'

                buffSize => the size of in-Memory buffer; default=1,000 feature vectors/images"""
        # Raise warning if outputPath exists
        if os.path.exists(outputPath):
            raise ValueError("The 'outputPath' supplied already exists and cannot be overwritten. Manually delete"
                             "the files to continue", outputPath)

        # open HDF5 database for writing data and create 2 datasets:
        # one for images/features and another for class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # store and initialize the buffer and index
        self.buffSize = buffSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        """Method: add
                This method is used to add data to buffer.

                rows => Number of rows to be added to the dataset

                labels => Corresponding class labels of added rows to the dataset"""
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.buffSize:
            self.flush()

    def flush(self):
        """Method: flush
        This method writes the current buffer to disk and resets the buffer"""
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        """Method: storeClassLabels
        This method will store the raw string names of the class labels in a separate HDF5 dataset."""
        dt = h5py.string_dtype()
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        """Method: close
        This method may be used to write data left in the buffer to the HDF5 database file when closed"""
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()

# _________________________________________________ END OF FILE __________________________________________________ #
