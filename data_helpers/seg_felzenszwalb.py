import cv
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap
import skimage.segmentation
import skimage.io
import skimage.color
import matplotlib.pylab as plt

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--directory', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--imfile', type=int, help='The number of image in NYU dataset.') 
    args = parser.parse_args()
    dataset_file = args.directory + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (args.imfile, args.imfile))
    print('Extracting the file' + dataset_file)
    im = skimage.color.rgb2gray(skimage.io.imread(dataset_file))
    seg_mask = skimage.segmentation.felzenszwalb(im, scale=100)
    normalized_seg = seg_mask.astype(np.float)/np.amax(seg_mask)*255
    plt.imshow(normalized_seg, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
