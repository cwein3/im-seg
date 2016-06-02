""" 
Given the pickle dataset, we will train a neural network to classify the HOG features.
"""
import cv
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap
import skimage.io
import skimage.color
import skimage.feature
import matplotlib.pylab as plt
import cPickle as pickle
from sknn.mlp import Classifier, Layer
from collections import Counter
import random
import sklearn.metrics

def fit_new(self, X, y, num_classes):
    """
    We have to rewrite the scikit-learn fit function so it doesn't do label binarization in that way.
    """
    assert X.shape[0] == y.shape[0],\
        "Expecting same number of input and output samples."
    if y.ndim == 1:
        y = y.reshape((y.shape[0], 1))

    if y.shape[1] == 1 and self.layers[-1].type != 'Softmax':
        log.warning('{}WARNING: Expecting `Softmax` type for the last layer '
                    'in classifier.{}\n'.format(ansi.YELLOW, ansi.ENDC))
    if y.shape[1] > 1 and self.layers[-1].type != 'Sigmoid':
        log.warning('{}WARNING: Expecting `Sigmoid` as last layer in '
                    'multi-output classifier.{}\n'.format(ansi.YELLOW, ansi.ENDC))

    yp = np.zeros((y.shape[0], num_classes))
    yp[np.arange(y.shape[0]), y] = 1

    # Also transform the validation set if it was explicitly specified.
    if self.valid_set is not None:
        X_v, y_v = self.valid_set
        if y_v.ndim == 1:
            y_v = y_v.reshape((y_v.shape[0], 1))
        with self._patch_sklearn():
            ys = [lb.transform(y_v[:,i]) for i, lb in enumerate(self.label_binarizers)]
        y_vp = numpy.concatenate(ys, axis=1)
        self.valid_set = (X_v, y_vp)

    # Now train based on a problem transformed into regression.
    return super(Classifier, self)._fit(X, yp, w)

def convert_data(data_loc, class_map):
    data_arr = pickle.load(open(data_loc, "r"))
    Xy_arr = []
    tot_examples = len(data_arr)
    allowed_num = np.inf
    if args.mode == 'TRAIN':
        allowed_num = tot_examples/args.num_classes if args.allowed_num is None else args.allowed_num
    class_counts = Counter()
    gc.disable()
    for ind, val in enumerate(data_arr): 
        class_name = val[3][0][0]   
        if class_name not in class_map:
            continue
        if class_counts[class_name] < allowed_num:
            class_counts[class_name] += 1
            Xy_arr.append((val[2], class_name))
        if ind % 100000 == 0: 
            print "Finished processing " + str(ind) + " patches."
    gc.enable()
    random.shuffle(Xy_arr)
    X = np.array([val[0] for val in Xy_arr])
    y = np.array([class_map[val[1]] for val in Xy_arr])
    
    print "Classes to indices map:", class_map
    print "Class counts:", class_counts 
    return X, y 

def train(X, y, num_classes, model=None):
    if model is None:
        model = Classifier(
            layers=[
                Layer("Sigmoid", units=args.num_hidden),
                Layer("Softmax", units=num_classes)], 
            learning_rule='sgd',
            learning_rate=args.lr,
            n_iter=args.n_iter,
            verbose=1)
    model.fit = fit_new
    model.fit(X, y, num_classes)
    pickle.dump(model, open(args.outfile, "w"))
    return model

def predict(X, y, model):
    labels = model.predict(X).flatten()
    n_samples = X.shape[0]
    err = float(np.sum(y != labels))/n_samples
    print "Prediction error of ", err, "."
    cm = sklearn.metrics.confusion_matrix(y, labels)
    row_sum = cm.sum(axis=1).reshape(cm.shape[0], 1)
    print "Frequencies of each class:", row_sum
    cm = cm.astype(float)/row_sum
    plt.matshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--data', type=str, help='The location of the data file we are using.')
    parser.add_argument('--mode', type=str, help='Either PREDICT or TRAIN. Will predict on test dataset or train on new dataset accordingly.')
    parser.add_argument('--old_model', type=str, default=None, help='The file location of the neural network model. If this is None, we will train a model from scratch, but this needs to be specified for predict.')
    parser.add_argument('--num_hidden', type=int, default=1000, help='The number of hidden layers in the classifier.')
    parser.add_argument('--n_iter', type=int, help='Number of iterations of gradient descent to use.')
    parser.add_argument('--outfile', type=str, help='The file which we output the trained model to.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate to use.')
    parser.add_argument('--allowed_num', type=int, help='The allowed number of training examples for one class.')
    parser.add_argument('--name_map', type=str, default='classmap.pkl', help='Name of the pickle file which stores the class map.')
    parser.add_argument('--num_split', type=int, default='13', help='The number of training splits to do.')
    global args
    args = parser.parse_args()
    class_map = pickle.load(open(args.name_map, "r"))
    num_classes = len(class_map)
    model = None if args.old_model is None else pickle.load(open(args.old_model, "r"))
    if args.mode == 'TRAIN':
        for split in xrange(args.num_split):
            data_loc = ("split%d" % split) + args.data
            X, y = convert_data(data_loc, class_map)
            model = train(X, y, num_classes, model) 
    if args.mode == 'PREDICT': 
        data_loc = ("split%d" % args.num_split) + args.data
        X, y = convert_data(data_loc, class_map)
        predict(X, y, model)
    
if __name__ == "__main__":
    main()
