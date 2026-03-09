# An MNIST data loader that splits data into
# training, validation and test sets.

import os
import numpy as np
import gzip
import struct

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

TRAIN_IMAGES = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
TRAIN_LABELS = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")
TEST_IMAGES = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, n_images, columns, rows = struct.unpack(">IIII", f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)



def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

def standardize(training_set, test_set):
    average = np.average(training_set)
    standard_deviation = np.std(training_set)
    training_set_standardized = (training_set - average) / standard_deviation
    test_set_standardized = (test_set - average) / standard_deviation
    return training_set_standardized, test_set_standardized

# X_train/X_validation/X_test: 60K/5K/5K images
# Each image has 784 elements (28 * 28 pixels)
X_train_raw = load_images(TRAIN_IMAGES)
X_test_all_raw = load_images(TEST_IMAGES)

# Standardize image data
X_train, X_test_all = standardize(X_train_raw, X_test_all_raw)
X_validation, X_test = np.split(X_test_all, 2)

# 60K labels, each a single digit from 0 to 9
Y_train_unencoded = load_labels(TRAIN_LABELS)

# Y_train: 60K labels, each consisting of 10 one-hot-encoded elements
Y_train = one_hot_encode(Y_train_unencoded)

# Y_validation/Y_test: 5K/5K labels, each a single digit from 0 to 9
Y_test_all = load_labels(TEST_LABELS)
Y_validation, Y_test = np.split(Y_test_all, 2)
