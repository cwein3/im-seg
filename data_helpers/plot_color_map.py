
import cv
import cv2
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap
import itertools
import skimage.io
import skimage.segmentation
import cPickle as pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.colors
import random
import pdb
import matplotlib.patches as mpatches

color_map = pickle.load(open("../crf/list_map.pkl", "r"))
allowed_classes = {u'bed':0, u'bookshelf':1, u'cabinet':2, u'ceiling':3, u'floor':4, u'picture':5, u'sofa':6, u'table':7, u'television':8, u'wall':9, u'window':10}
inv_map = {v : k for k, v in allowed_classes.items()}

recs = []
classes = ['bed', 'bookshelf', 'cabinet', 'ceiling', 'floor', 'picture', 'sofa', 'table', 'television', 'wall', 'window']

im_blah = np.arange(len(classes)).reshape((len(classes), 1))
plt.imshow(im_blah, cmap=color_map, interpolation='None')
plt.yticks(np.arange(len(classes)), classes, rotation='horizontal')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.colorbar(ticks=range(len(classes)))
plt.show()
for ind, _ in enumerate(classes):
    recs.append(mpatches.Rectangle((0, 0),1,1,fc=color_map[ind]))


plt.legend(recs, classes)
plt.show()
