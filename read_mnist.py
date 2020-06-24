import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip
def load_mnist():
    # Import MNIST data
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    # Training data, only
    X = valid_set[0]
    y = valid_set[1]
  
    # change y [1D] to Y [2D] sparse array coding class
    n_examples = len(y)
    labels = np.unique(y)
    Y = np.zeros((n_examples, len(labels)))
    for ix_label in range(len(labels)):
        # Find examples with with a Label = lables(ix_label)
        ix_tmp = np.where(y == labels[ix_label])[0]
        Y[ix_tmp, ix_label] = 1
    return X, Y, labels, y


