import numpy as np
def load_car():
    # Import MNIST data
    fac = 0.99 / 4
    data = np.loadtxt("CarBinary.csv",delimiter=",",max_rows=1000)
    # Training data, only
    X = np.asfarray(data[:, 1:7]) * fac + 0.01
    y = [int(x[0]) for x in np.asfarray(data[:, 7:])]
    # change y [1D] to Y [2D] sparse array coding class
    n_examples = len(y)
    labels = np.unique(y)
    Y = np.zeros((n_examples, len(labels)))
    for ix_label in range(len(labels)):
        # Find examples with with a Label = lables(ix_label)
        ix_tmp = np.where(y == labels[ix_label])[0]
        Y[ix_tmp, ix_label] = 1
    return X, Y, labels, y

