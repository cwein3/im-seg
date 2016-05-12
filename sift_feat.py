import cv
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--directory', type=string, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--imfile', type=int, help='The number of image in NYU dataset.') 
    args = parser.parse_args()
    dataset_file = args.directory + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (args.imfile, args.imfile))
    print('Extracting the file' + dataset_file)
    im = scipy.misc.imread(dataset_file)
    
