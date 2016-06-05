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
import types

allowed_classes = {u'bed':0, u'bookshelf':1, u'cabinet':2, u'ceiling':3, u'floor':4, u'picture':5, u'sofa':6, u'table':7, u'television':8, u'wall':9, u'window':10}

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
    # jankify
    curr_len = len(Xy_arr)
    for key in class_map:
	if class_counts[key] == 0:
	    class_counts[key] += 1
	    Xy_arr.append((np.zeros(Xy_arr[-1][0].shape), key))
    gc.enable()
    w = np.ones((len(Xy_arr),))
    w[curr_len:] = 0
    random.shuffle(Xy_arr)
    X = np.array([val[0] for val in Xy_arr])
    y = np.array([class_map[val[1]] for val in Xy_arr])
    
    print "Classes to indices map:", class_map
    print "Class counts:", class_counts 
    return X, y, w 

def train(X, y, w, num_classes, model=None, lr=0.01):
    if model is None:
        model = Classifier(
            layers=[
                Layer("Sigmoid", units=args.num_hidden),
                Layer("Softmax", units=num_classes)], 
            learning_rule='sgd',
            learning_rate=lr,
            n_iter=1,
            verbose=1)
    model.fit(X, y)#, w=w)
    pickle.dump(model, open(args.outfile, "w"))
    labels = model.predict(X).flatten()
    print "Split accuracy", float(np.sum(labels == y))/X.shape[0]
    return model

def predict_split(X, y, model):
    labels = model.predict(X).flatten()
    n_samples = X.shape[0]
    print "Split accuracy", float(np.sum(labels == y))/n_samples
    return labels

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data_helpers/', help='Folder containing data splits.')
    parser.add_argument('--data', type=str, help='The location of the data file we are using.')
    parser.add_argument('--mode', type=str, help='Either PREDICT or TRAIN. Will predict on test dataset or train on new dataset accordingly.')
    parser.add_argument('--old_model', type=str, default=None, help='The file location of the neural network model. If this is None, we will train a model from scratch, but this needs to be specified for predict.')
    parser.add_argument('--num_hidden', type=int, default=1000, help='The number of hidden layers in the classifier.')
    parser.add_argument('--n_iter', type=int, default=2, help='Number of iterations of gradient descent to use.')
    parser.add_argument('--outfile', type=str, help='The file which we output the trained model to.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate to use.')
    parser.add_argument('--allowed_num', type=int, help='The allowed number of training examples for one class.')
    parser.add_argument('--name_map', type=str, default='classmap.pkl', help='Name of the pickle file which stores the class map.')
    parser.add_argument('--num_split', type=int, default='13', help='The number of training splits to do.')
    parser.add_argument('--predict_set', type=str, default='test', help='Whether we predict on train or test.')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='Learning rate decay every time we pass over a split.')
    parser.add_argument('--hardcode', type=bool, default=False, help='Whether to hardcode allowed number of classes.')
    global args
    args = parser.parse_args()
    class_map = allowed_classes if args.hardcode else pickle.load(open(args.name_map, "r"))
    num_classes = len(class_map)
    model = None if args.old_model is None else pickle.load(open(args.old_model, "r"))
    if args.mode == 'TRAIN':
	lr = args.lr
	for _ in xrange(args.n_iter):
            for split in xrange(args.num_split):
                data_loc = args.data_dir + ("trainsplit%d" % split) + args.data
                X, y, w = convert_data(data_loc, class_map)
                model = train(X, y, w, num_classes, model, lr)
		lr *= args.lr_decay 
    if args.mode == 'PREDICT':
	all_predict = np.array([])
	all_labels = np.array([])
	for split in xrange(args.num_split):
	    data_loc = args.data_dir + (args.predict_set + "split%d" % split) + args.data
            X, y, _ = convert_data(data_loc, class_map)
            all_predict = np.concatenate((all_predict, predict_split(X, y, model)), axis=0)
	    all_labels = np.concatenate((all_labels, y), axis=0)
       	n_samples = all_labels.size
       	err = float(np.sum(all_predict != all_labels))/n_samples
       	print "Prediction error of ", err, "."
       	cm = sklearn.metrics.confusion_matrix(all_labels, all_predict)
       	row_sum = cm.sum(axis=1).reshape(cm.shape[0], 1)
       	print "Frequencies of each class:", row_sum
       	cm = cm.astype(float)/row_sum
       	plt.matshow(cm)
       	plt.title("Confusion Matrix")
       	plt.colorbar()
       	plt.show()
       	   
       
if __name__ == "__main__":
    main()
