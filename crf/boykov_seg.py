"""
Given a the appearance model and the image, we first segment into rough superpixels. Then we assign class probabilities for these superpixels, and using this we find the MAP assignment of the conditional random field.
"""
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
import gco_python.pygco 

colors = [(random.random(),random.random(),random.random()) for i in xrange(256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

def assign_superpix_prob(im, network, descs):
    """
    im: image, assumed to be in grayscale
    network: neural network for appearance classification
    descs: list of tuple of (desc_x, desc_y, sift feature)
    """
    im = skimage.color.rgb2gray(im)
    seg_mask = skimage.segmentation.felzenszwalb(im, scale=100)
    superpix_probs = {} # map a superpixel to the average of probabilities for descriptors falling in it
    superpix_counter = collections.Counter()
    predict_descs = np.array([desc[2] for desc in descs])
    probs = network.predict_proba(predict_descs)
    
    for ind, desc in enumerate(descs): 
        desc_x = desc[0]
        desc_y = desc[1]
        superpix = seg_mask[desc_y, desc_x]
        if superpix in superpix_probs:
            superpix_probs[superpix] += probs[ind]
        else:
            superpix_probs[superpix] = probs[ind].copy()
        superpix_counter[superpix] += 1
    
    for superpix in superpix_probs:
        superpix_probs[superpix] /= superpix_counter[superpix]
    
    num_classes = probs.shape[1]
    num_super = np.amax(seg_mask)

    for superpix in xrange(num_super + 1):
        if superpix not in superpix_probs:
            superpix_probs[superpix] = 1.0/num_super*np.ones((num_classes,))

    return seg_mask, superpix_probs

def calc_pix_grad(im):
    """
    Not really sure how to do this so we'll leave it at this for now.
    im: image in RGB
    """
    H, W, _ = im.shape
    pix_grad_vert = np.sum((im[:H - 1, :, :] - im[1:, :, :])**2, axis=2)
    pix_grad_hor = np.sum((im[:, :W - 1, :] - im[:, 1:, :])**2, axis=2)
    return pix_grad_vert, pix_grad_hor

def boykov(im, crf_params, loc_probs, plot_every=1, live_plot=False):
    d, eta0, alpha, t = crf_params
    seg_mat, superpix_probs = loc_probs
    H = im.shape[0]
    W = im.shape[1]
    num_classes = superpix_probs[0].shape[0]
    costs = np.zeros((W, H, num_classes), dtype=np.int32)
    probs_arr = [None for _ in xrange(len(superpix_probs))]

    for key in superpix_probs:
        probs_arr[key] = -np.log(superpix_probs[key])

    for h in xrange(H):
        for w in xrange(W):
            costs[w, h] = (100*probs_arr[seg_mat[h, w]]).astype(np.int32)

    pairwise_cost = (d - d*np.eye(num_classes)).astype(np.int32)    
    grad_vert, grad_hor = calc_pix_grad(im)
    grad_vert = eta0*np.exp(-alpha*grad_vert)*100
    grad_hor = eta0*np.exp(-alpha*grad_hor)*100
    grad_vert = np.concatenate((grad_vert, np.zeros((1,W))), axis=0).astype(np.int32)
    grad_hor = np.concatenate((grad_hor, np.zeros((H,1))), axis=1).astype(np.int32)
    grad_vert_T = np.zeros((W, H), dtype=np.int32)
    grad_hor_T = np.zeros((W, H), dtype=np.int32)
    grad_vert_T[:] = grad_vert.T
    grad_hor_T[:] = grad_hor.T
   
    print grad_vert_T, grad_hor_T 
    lab_assign = gco_python.pygco.cut_simple_vh(costs, pairwise_cost, grad_vert_T, grad_hor_T, algorithm='expansion')
    return lab_assign

def segment_im(imfile, featfile, network_file, d, eta0, alpha, t):
    feat_f = open(featfile, "r")
    im = skimage.io.imread(imfile)    
    network = pickle.load(open(network_file, "r"))
    descs = []
        
    for line in feat_f: 
        parts = line.split(" ")
        centers = (int(float(parts[0])), int(float(parts[1])))
        feat = np.array(map(float, filter(bool, parts[2].strip().split(','))))
        descs.append((centers[0], centers[1], feat))
    
    seg_mask, pix_probs = assign_superpix_prob(im, network, descs)
    assign = boykov(im, (d, eta0, alpha, t), (seg_mask, pix_probs), plot_every=1, live_plot=False)
    feat_f.close()    
    return assign

def find_acc(filenum, network_file):
    mat_path = args.SUN_dir + ('SUNRGBD/kv1/NYUdata/NYU%04d/seg.mat' % filenum)
    seg_labels = scipy.io.loadmat(mat_path)['seglabel']
    names = scipy.io.loadmat(mat_path)['names']
    imfile = args.SUN_dir + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (args.imfile, args.imfile))
    featfile = args.feat_dir + ('%04d.txt' % args.imfile)
    network_file = args.network_path
    assign = segment_im(imfile, featfile, network_file, args.d, args.eta0, args.alpha, args.t)
    assign = assign.T
    plt.imshow(assign, cmap=new_map)
    plt.show()
    class_map = pickle.load(open(args.class_map, "r"))
    conv_labels = np.zeros(assign.shape)
    for h in xrange(assign.shape[0]):
        for w in xrange(assign.shape[1]):
            curr_name = names[seg_labels[h, w] - 1][0][0]
            if curr_name not in class_map:
                class_map[curr_name] = len(class_map)
            conv_labels[h, w] = class_map[curr_name]

    numpix = assign.shape[0]*assign.shape[1]
    acc_map = (conv_labels == assign)
    plt.imshow(acc_map, cmap=new_map)
    plt.show()

    print "Pixel accuracy:", float(np.sum(acc_map))/numpix

def main():
    global args
    parser = ap.ArgumentParser()
    parser.add_argument('--SUN_dir', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--feat_dir', type=str, help='The directory where all the HOG features are stored.') 
    parser.add_argument('--network_path', type=str, help='Path of the neural network we use.')
    parser.add_argument('--imfile', type=int, help='Which image we decide to use.')
    parser.add_argument('--mode', type=str, default='SEGMENT', help='What function to call.')
    parser.add_argument('--d', type=float, default=1, help='Parameter in the crf, see NYU paper 1.')
    parser.add_argument('--eta0', type=float, default=10, help='See first NYU paper for parameter description.')
    parser.add_argument('--alpha', type=float, default=10, help='See first NYU paper for parameter description.')
    parser.add_argument('--t', type=float, default=0, help='See first NYU paper for parameter description.')
    parser.add_argument('--class_map', type=str, default='../classify/classmap.pkl', help='Class map.')
    args = parser.parse_args()
    imfile = args.SUN_dir + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (args.imfile, args.imfile))
    featfile = args.feat_dir + ('%04d.txt' % args.imfile)
    network_file = args.network_path
    if args.mode == 'SEGMENT':
        find_acc(args.imfile, network_file)
    if args.mode == 'GRADS':
        plot_pix_grads(imfile)

if __name__ == '__main__':
    main()
