import cv
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap
import skimage.segmentation
import skimage.io
import skimage.color
import matplotlib.pylab as plt
import matplotlib.colors
import random

colors = [(1,1,1)] + [(random.random(),random.random(),random.random()) for i in xrange(255)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--directory', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--imfile', type=int, help='The number of image in NYU dataset.') 
    args = parser.parse_args()
    dataset_file = args.directory + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (args.imfile, args.imfile))
    print('Extracting the file' + dataset_file)
    im = skimage.io.imread(dataset_file)
    seg_mask = skimage.segmentation.slic(im)
    normalized_seg = seg_mask.astype(np.float)/np.amax(seg_mask)*255
    plt.imsave("slic.jpg", normalized_seg, cmap=new_map)

if __name__ == '__main__':
    main()
