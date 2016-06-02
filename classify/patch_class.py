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
import gc

allowed_classes = {u'bed', u'blind', u'bookshelf', u'cabinet', u'ceiling', u'floor', u'picture', u'sofa', u'table', u'television', u'wall', u'window'}

def convert_data():
    data_arr = pickle.load(open(args.data, "r"))
    Xy_arr = []
    class_map = {} if args.name_map is None else pickle.load(open(args.name_map, "r"))
    tot_examples = len(data_arr)
    allowed_num = np.inf
    if args.mode == 'TRAIN':
        allowed_num = tot_examples/args.num_classes if args.allowed_num is None else args.allowed_num
    class_counts = Counter()
    gc.disable()
    for ind, val in enumerate(data_arr): 
        class_name = val[3][0][0]   
        if class_name not in allowed_classes:
	    continue
	if class_counts[class_name] < allowed_num:
            class_counts[class_name] += 1
            Xy_arr.append((val[2], class_name))
        if class_name not in class_map:
            class_map[class_name] = len(class_map)
        if ind % 100000 == 0: 
            print "Finished processing " + str(ind) + " patches."
    gc.enable()
    random.shuffle(Xy_arr)
    num_classes = len(class_map)
    X = np.array([val[0] for val in Xy_arr])
    y = np.array([class_map[val[1]] for val in Xy_arr])
    
    if args.name_map is None:
    	pickle.dump(class_map, open(args.new_name_map, "w"))

    print "Classes to indices map:", class_map
    print "Class counts:", class_counts 
    return X, y, num_classes    

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
    model.fit(X, y)
    pickle.dump(model, open(args.outfile, "w"))
    
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
    parser.add_argument('--num_classes', type=int, default=110, help='Number of total classes.')
    parser.add_argument('--allowed_num', type=int, help='The allowed number of training examples for one class.')
    parser.add_argument('--name_map', type=str, default=None, help='Name of the pickle file which stores the class map.')
    parser.add_argument('--new_name_map', type=str, default='classmap.pkl', help='Name of pickle file we dump class map to.')
    global args
    args = parser.parse_args()
    X, y, num_classes = convert_data()
    model = None if args.old_model is None else pickle.load(open(args.old_model, "r"))
    if args.mode == 'PREDICT': 
        predict(X, y, model)
    if args.mode == 'TRAIN':
        train(X, y, num_classes, model)

if __name__ == "__main__":
    main()

