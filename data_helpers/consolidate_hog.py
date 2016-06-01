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
import cPickle as pickle
import gc

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
        seg_label = names[seg_labels[int(centers[1]) - 1, int(centers[0]) - 1] - 1]
        dat_list.append((filenum, centers, HOG_part, seg_label))
    HOG_file.close()
    return dat_list

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--SUN_dir', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--HOG_dir', type=str, help='The directory where all the HOG features are stored.') 
    parser.add_argument('--num_files', type=int, help='The number of .txt files in the directory containing the HOG features.')
    parser.add_argument('--outfile', type=str, help='The name of the pickle file we output the dataset to.')
    parser.add_argument('--num_split', type=int, help='The number of splits to create. The last split will be test. Other splits are so we do not have to load huge file into memory.')
    global args
    args = parser.parse_args()
    num_split = args.num_split
    for i in xrange(num_split):
        gc.disable()
        tot_data = []
        for filenum in xrange(i*args.num_files/num_split + 1, (i + 1)*(args.num_files/num_split)+ 1): 
            tot_data += single_file_extract(filenum)
            if filenum % 100 == 0: 
                print "Finished processing " + str(filenum) + " files."
            pickle.dump(tot_data, open(("split%d" % i) + args.outfile, "w"))
        gc.enable()

if __name__ == "__main__":
    main()
