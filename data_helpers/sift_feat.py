import cv
import cv2
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap
import itertools

def extract_sift_feat(filename, imnum):
    im = cv2.imread(filename)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    extractor = cv2.DescriptorExtractor_create("SIFT")
    num_h = im.shape[0]/args.stride + 1 
    num_w = im.shape[1]/args.stride + 1
    kps = []
    for h in xrange(num_h):
        for w in xrange(num_w):
            h_start = h*args.stride
            h_end = h*args.stride + args.patch_size
            w_start = w*args.stride
            w_end = w*args.stride + args.patch_size
            if h_end > gray.shape[0] or w_end > gray.shape[1]:
                continue
            h_mid = (h_start + h_end)/2
            w_mid = (w_start + w_end)/2
            kps.append(cv2.KeyPoint(w_mid, h_mid, args.keypoint_size))
    kps, descs = extractor.compute(gray, kps)
    outfile = open(args.output_dir + ('%04d.txt' % imnum), "w")
    for kp, desc in itertools.izip(kps, descs):
        centerw = kp.pt[0]
        centerh = kp.pt[1]
        outfile.write(str(centerw) + " " + str(centerh) + " ")
        cv2.normalize(desc, desc)
        desc.tofile(outfile, sep=",", format="%f")
        outfile.write("\n")
    outfile.close()
    
def main():
    global args 
    parser = ap.ArgumentParser()
    parser.add_argument('--directory', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--numfiles', type=int, help='The number of images in NYU dataset.') 
    parser.add_argument('--stride', type=int, default=10, help='Stride of patches for which we extract HOG features.')
    parser.add_argument('--patch_size', type=int, default=40, help='Side length of path size for which we extract HOG features.')   
    parser.add_argument('--keypoint_size', type=float, default=1.0, help='Size of the keypoint.')
    parser.add_argument('--output_dir', type=str, help='Directory to output the sift features.')
    args = parser.parse_args()
    for imfile in xrange(1, args.numfiles + 1):
        dataset_file = args.directory + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (imfile, imfile))
        extract_sift_feat(dataset_file, imfile)
        if imfile % 100 == 0: 
            print "Finished " + str(imfile) + " files."

if __name__ == '__main__':
    main()
