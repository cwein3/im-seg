import cv
import cv2
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap

def extract_sift_feat(filename):
    im = cv2.imread(filename)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sift_detector = cv2.FeatureDetector_create("SIFT")
    extractor = cv2.DescriptorExtractor_create("SIFT")
    num_h = im.shape[0]/args.stride + 1 
    num_w = im.shape[1]/args.stride + 1
    for h in xrange(num_h):
        for w in xrange(num_w):
            h_start = h*args.stride
            h_end = h*args.stride + args.patch_size
            w_start = w*args.stride
            w_end = w*args.stride + args.patch_size
            if h_end > gray.shape[0] or w_end > gray.shape[1]:
                continue
            gray_slice = gray[h_start:h_end, w_start:w_end]
            kps = sift_detector.detect(gray_slice)
            kps, descs = extractor.compute(gray_slice, kps)
            print kps, descs

def main():
    global args 
    parser = ap.ArgumentParser()
    parser.add_argument('--directory', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--imfile', type=int, help='The number of image in NYU dataset.') 
    parser.add_argument('--stride', type=int, default=10, help='Stride of patches for which we extract HOG features.')
    parser.add_argument('--patch_size', type=int, default=40, help='Side length of path size for which we extract HOG features.')   
    args = parser.parse_args()
    dataset_file = args.directory + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (args.imfile, args.imfile))
    extract_sift_feat(dataset_file)
    
if __name__ == '__main__':
    main()
