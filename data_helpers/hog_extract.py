import cv
import numpy as np
import scipy.io
import scipy.misc
import argparse as ap
import skimage.io
import skimage.color
import skimage.feature
import matplotlib.pylab as plt

def feat_for_image(filename, imnum):
    stride = args.stride
    im = skimage.color.rgb2gray(skimage.io.imread(filename))
    num_h = im.shape[0]/stride + 1 
    num_w = im.shape[1]/stride + 1
    
    # we want the descriptor size to be approximately 128
    num_cells = 128/args.orientations + 1
    cells_per = np.ceil(np.sqrt(num_cells)).astype(int)
    cell_size = np.ceil(args.patch_size/cells_per).astype(int)

    # plt.imshow(im, cmap="gray")
    assert(args.output_dir[-1] == '/')
    outfile = open(args.output_dir + ('%04d.txt' % imnum), "wb")
    
    for h in xrange(num_h):
        for w in xrange(num_w):
            h_start = h*stride
            h_end = h*stride + args.patch_size
            w_start = w*stride
            w_end = w*stride + args.patch_size
            if h_end > im.shape[0] or w_end > im.shape[1]:
                continue
            im_slice = im[h_start:h_end, w_start:w_end]
            hog = skimage.feature.hog(im_slice, args.orientations, pixels_per_cell=(cell_size, cell_size), cells_per_block=(cells_per, cells_per))
            # plt.plot([(w_start + w_end)/2], [(h_start + h_end)/2], marker='v')
            centerw = (w_start + w_end)/2
            centerh = (h_start + h_end)/2
            outfile.write(str(centerw) + " " + str(centerh) + " ")
            hog.tofile(outfile, sep=",", format="%f")
            outfile.write("\n")
    outfile.close()        
    # plt.show()

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--directory', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--numfiles', type=int, help='The number of images in NYU dataset.') 
    parser.add_argument('--orientations', type=int, default=9, help='Number of HOG orientations we bin into.')
    parser.add_argument('--stride', type=int, default=10, help='Stride of patches for which we extract HOG features.')
    parser.add_argument('--patch_size', type=int, default=40, help='Side length of path size for which we extract HOG features.')
    parser.add_argument('--output_dir', type=str, help='Where to output the extracted HOG features.')
    global args
    args = parser.parse_args()
    for imfile in xrange(1, args.numfiles + 1):
        dataset_file = args.directory + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (imfile, imfile))
        feat_for_image(dataset_file, imfile)
        if imfile % 100 == 0: 
            print "Finished " + str(imfile) + " files."

if __name__ == "__main__":
    main()
