""" 
For given HOG features saved in a .txt dataset, this file will convert to a list of tuple of (img num, center of patch, HOG descriptor, and segmentation pixel label). Assume that the .txt dataset file names are written as <filenum>.txt. 
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
import random
import cPickle as pickle
import gc
import collections

class_counts = collections.Counter()

def single_file_extract(filenum): 
    HOG_file = open(args.HOG_dir + ("%04d.txt" % filenum), "r")
    mat_path = args.SUN_dir + ('SUNRGBD/kv1/NYUdata/NYU%04d/seg.mat' % filenum)
    seg_labels = scipy.io.loadmat(mat_path)['seglabel']
    names = scipy.io.loadmat(mat_path)['names']
    dat_list = []
    for line in HOG_file: 
        parts = line.split(" ")
        centers = (float(parts[0]), float(parts[1]))
        HOG_part = np.array(map(float, filter(bool, parts[2].strip().split(','))))
        # seg_labels goes by y first, then x
	if seg_labels[int(centers[1]) -1, int(centers[0]) - 1] == 0: 
	    continue
        seg_label = names[seg_labels[int(centers[1]) - 1, int(centers[0]) - 1] - 1]
   	if not args.count_classes_only:
	    dat_list.append((filenum, centers, HOG_part, seg_label))
        class_counts[seg_label[0][0]] += 1
    HOG_file.close()
    return dat_list

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--SUN_dir', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--HOG_dir', type=str, help='The directory where all the HOG features are stored.') 
    parser.add_argument('--outfile', type=str, help='The name of the pickle file we output the dataset to.')
    parser.add_argument('--num_split', type=int, help='The number of splits to create. The last split will be test. Other splits are so we do not have to load huge file into memory.')
    parser.add_argument('--names_map', type=str, help='The name of the file we output the list of class names to.')
    parser.add_argument('--num_classes', type=int, help='The number of most common classes which we take.')
    parser.add_argument('--count_classes_only', type=bool, default=False, help='Whether or not to count the number of classes only.')
    parser.add_argument('--hardcode', type=bool, default=False, help='Whether or not to use hardcoded class map.')
    parser.add_argument('--feat_map', type=str, help='Where to store example features map.')
    global args
    args = parser.parse_args()
    num_split = args.num_split
    train_test = scipy.io.loadmat(args.SUN_dir + "splits.mat")
    for type in ['train', 'test']:
        inds = train_test[type + 'Ndxs']
	num_files = len(inds)
        random.shuffle(inds)
	tot_data = []
	gc.disable()
	for filenum in xrange(num_files):
	    tot_data += single_file_extract(inds[filenum])
	    if filenum % 100 == 0:
		print "Finished processing " + str(filenum) + " files."
	random.shuffle(tot_data)
	gc.enable()
	num_per_split = len(tot_data)/num_split
	for i in xrange(num_split):
	    end_ind = min((i + 1)*num_per_split, len(tot_data))
	    if not args.count_classes_only:
	        pickle.dump(tot_data[i*num_per_split:end_ind], open(type + ("split%d" % i) + args.outfile, "w"))
        
    top_dict = dict(class_counts.most_common(args.num_classes))
    print "Counts of different classes:", top_dict
    count_key = {}
    for key in top_dict:
        count_key[key] = len(count_key)
    pickle.dump(count_key, open(args.names_map, "w"))

if __name__ == "__main__":
    main()
