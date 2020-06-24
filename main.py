import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip
import read_car as car
import read_mnist as mnist
import read_iris as iris

class Mlp():
    def __init__(self, size_layers, reg_lambda=0, bias_flag=True):
        self.size_layers = size_layers
        self.n_layers    = len(size_layers)
        self.lambda_r    = reg_lambda
        self.bias_flag   = bias_flag
        self.initialize_theta_weights()#randomly

    def train(self, X, Y, iterations=400, reset=False):
        n_examples = Y.shape[0]
        if reset:
            self.initialize_theta_weights()
        for iteration in range(iterations):
            self.gradients = self.backpropagation(X, Y)
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.theta_weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.theta_weights = self.roll_weights(self.theta_vector)

    def predict(self, X):
        A , Z = self.feedforward(X)
        Y_hat = A[-1]
        return Y_hat

    def initialize_theta_weights(self):
        self.theta_weights = []
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            epsilon = 4.0 * np.sqrt(6) / np.sqrt(size_layer + size_next_layer)
            if self.bias_flag: # Weigts from a uniform distribution [-epsilon, epsion] 
                theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer + 1) * 2.0 ) - 1)
            else:
                theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer) * 2.0 ) - 1)
            self.theta_weights.append(theta_tmp)
        return self.theta_weights

    def backpropagation(self, X, Y):
        g_dz = lambda x: self.sigmoid_derivative(x)
        n_examples = X.shape[0]
        A, Z = self.feedforward(X)# Feedforward        
        deltas = [None] * self.n_layers# Backpropagation
        deltas[-1] = A[-1] - Y
        for x_layer in np.arange(self.n_layers - 1 - 1 , 0 , -1):# For the second last layer to the second one
            theta_tmp = self.theta_weights[x_layer]
            if self.bias_flag:# Removing weights for bias
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[x_layer] = (np.matmul(theta_tmp.transpose(), deltas[x_layer + 1].transpose() ) ).transpose() * g_dz(Z[x_layer])
        gradients = [None] * (self.n_layers - 1) # Compute gradients
        for x_layer in range(self.n_layers - 1):
            grads_tmp = np.matmul(deltas[x_layer + 1].transpose() , A[x_layer])
            grads_tmp = grads_tmp / n_examples
            if self.bias_flag:
                grads_tmp[:, 1:] = grads_tmp[:, 1:] + (self.lambda_r / n_examples) * self.theta_weights[x_layer][:,1:] # Regularize weights, except for bias weigths
            else:
                grads_tmp = grads_tmp + (self.lambda_r / n_examples) * self.theta_weights[x_layer] # Regularize ALL weights      
            gradients[x_layer] = grads_tmp;
        return gradients

    def feedforward(self, X):
        g = lambda x: self.sigmoid(x)
        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = X
        for x_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            if self.bias_flag:
                input_layer = np.concatenate((np.ones([n_examples ,1]) ,input_layer), axis=1)# Add bias element to every example in input_layer
            A[x_layer] = input_layer
            Z[x_layer + 1] = np.matmul(input_layer,  self.theta_weights[x_layer].transpose() ) # Multiplying input_layer by theta_weights for this layer
            output_layer = g(Z[x_layer + 1]) # Activation Function
            input_layer = output_layer # Current output_layer will be next input_layer
        A[self.n_layers - 1] = output_layer
        return A, Z

    def unroll_weights(self, rolled_data):
        unrolled_array = np.array([])
        for one_layer in rolled_data:
            unrolled_array = np.concatenate((unrolled_array, one_layer.flatten('F')) )
        return unrolled_array

    def roll_weights(self, unrolled_data):
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        rolled_list = []
        if self.bias_flag:
            extra_item = 1
        else:
            extra_item = 0
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            n_weights = (size_next_layer * (size_layer + extra_item))
            data_tmp = unrolled_data[0 : n_weights]
            data_tmp = data_tmp.reshape(size_next_layer, (size_layer + extra_item), order = 'F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
        return rolled_list

    def sigmoid(self, z):
        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def relu(self, z):
        if np.isscalar(z):
            result = np.max((z, 0))
        else:
            zero_aux = np.zeros(z.shape)
            meta_z = np.stack((z , zero_aux), axis = -1)
            result = np.max(meta_z, axis = -1)
        return result

    def sigmoid_derivative(self, z):
        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result

    def relu_derivative(self, z):
        result = 1 * (z > 0)
        return result
    
def initialize(size,epochs,dset,reg=0):
    if dset == 1:
        size = [784]+size+[10]
        X, Y, labels, y  = mnist.load_mnist()
    elif dset == 3:
        size = [4]+size+[3]
        X, Y, labels, y  = iris.load_iris()
        mlp_classifier = Mlp(size,reg_lambda = reg)
    elif dset == 2:
        size = [6]+size+[4]
        X, Y, labels, y  = car.load_car()
    cutoff = int(len(X)*0.8)
    mlp_classifier = Mlp(size,reg_lambda = reg)
    acc_v = np.zeros([epochs,1])#loss
    loss = np.zeros([epochs,1])#Itarate
    for x in range(epochs):
        mlp_classifier.train(X[:cutoff], Y[:cutoff], 1)
        #loss - aactual
        Y_hat = mlp_classifier.predict(X[:cutoff])
        loss[x] = (0.5)*np.square(Y_hat - Y[:cutoff]).mean()
        #loss - pointless
        Y_hat_v = mlp_classifier.predict(X[cutoff:])
        acc_v[x] = (0.5)*np.square(Y_hat_v - Y[cutoff:]).mean()
        if x > 10 and acc_v[x-1] < acc_v[x]:
            break
    # Ploting loss vs epochs
    plt.figure()
    cut = (loss == 0).sum()
    x = np.arange(epochs-cut)
    plt.plot(x, loss[:epochs-cut],'b-',label='Validation')
    plt.plot(x, acc_v[:epochs-cut],'r-',label='Error')
    plt.legend(loc='best')
    plt.ylabel('Error(/100)')
    plt.xlabel('Epoch')
    plt.savefig('graph.png')
    # Training Accuracy
    Y_hat = mlp_classifier.predict(X[cutoff:])
    y_tmp = np.argmax(Y_hat, axis=1)
    y_hat = labels[y_tmp]
    # Getting Accuracy
    acc = np.mean(1 * (y_hat == y[cutoff:]))
    return str(acc*100)

#print(initialize([13],400,2,0.0))
