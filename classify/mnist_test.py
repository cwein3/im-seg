import argparse
import collections
import IPython
import json
import math
from math import log, exp
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import sys
import scipy.optimize as opt
import scipy.io
import scipy.misc as msc
from sknn.mlp import Classifier, Layer
import sklearn.metrics

def load_mnist(directory):
    all_dirs = scipy.io.loadmat(os.path.join(directory, 'mnist_all.mat'))
    train_vars = [
        (i, 'train%d' % i)
        for i in xrange(10)]
    test_vars = [
        (i, 'test%d' % i)
        for i in xrange(10)]

    train_Xs = [(i, all_dirs[v].astype(np.float64)/255) for i, v in train_vars]
    test_Xs = [(i, all_dirs[v].astype(np.float64)/255) for i, v in test_vars]

    train_Ys = [np.ones((m.shape[0], 1)) * i for i, m in train_Xs]
    test_Ys = [np.ones((m.shape[0], 1)) * i for i, m in test_Xs]

    train_Y = np.vstack(train_Ys)
    test_Y = np.vstack(test_Ys)

    train_X = np.vstack([m for _, m in train_Xs])
    test_X = np.vstack([m for _, m in test_Xs])

    assert train_X.shape[0] == train_Y.shape[0]
    assert test_X.shape[0] == test_Y.shape[0]

    return train_X, train_Y, test_X, test_Y, train_Xs, train_Ys

def main():
    train_X, train_Y, test_X, test_Y, train_Xs, train_Ys = load_mnist(".")
    model = Classifier(
            layers=[
                Layer("Sigmoid", units=1000),
                Layer("Softmax", units=10)], 
            learning_rule='sgd',
            learning_rate=0.01,
            n_iter=10,
            verbose=1)
    #model.fit(train_X, train_Y)
    train_Y = train_Y.flatten()
    test_Y = test_Y.flatten()
    #pickle.dump(model, open("mnist_model.pkl", "w"))    
    model = pickle.load(open("mnist_model.pkl", "r"))
    labels_train = model.predict(train_X).flatten()   
    labels_test = model.predict(test_X).flatten()
    num_train = labels_train.shape[0]
    num_test = labels_test.shape[0]
    train_err = float(np.sum(labels_train != train_Y))/num_train
    test_err = float(np.sum(labels_test != test_Y))/num_test
    print "Training error:", train_err
    print "Test error:", test_err

if __name__ == '__main__':
    main()
