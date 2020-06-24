import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip
def load_iris():
    # Import iris data
    data = np.loadtxt("IrisBinary.csv",delimiter=",",max_rows=1000)
    # Training data, only
    X = np.asfarray(data[:, 1:5])
    X[:,0] *= 0.99 / 4
    X[:,1] *= 0.99 / 3
    X[:,2] *= 0.99 / 6
    X[:,3] *= 0.99 / 3
    y = [int(x[0]) for x in np.asfarray(data[:, 5:])]
    y = [x*0.99 / 3 for x in y]
    # change y [1D] to Y [2D] sparse array coding class
    n_examples = len(y)
    labels = np.unique(y)
    Y = np.zeros((n_examples, len(labels)))
    for ix_label in range(len(labels)):
        # Find examples with with a Label = lables(ix_label)
        ix_tmp = np.where(y == labels[ix_label])[0]
        Y[ix_tmp, ix_label] = 1
    return X, Y, labels, y

