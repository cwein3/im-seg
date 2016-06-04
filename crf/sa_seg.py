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

import pyximport
pyximport.install(
    setup_args={"include_dirs":np.get_include()}
    )

import sa_cy

colors = [(random.random(),random.random(),random.random()) for i in xrange(255)]
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
    
def calc_pix_grad(im, p1, p2):
    """
    Not really sure how to do this so we'll leave it at this for now.
    im: image in RGB
    p1, p2: tuples for the (y, x) of the image coords
    """
    #ret = np.abs(im[p1[0], p1[1]] - im[p2[0], p2[1]])
    ret = np.exp(-0.001*np.sum((im[p1[0], p1[1], :] - im[p2[0], p2[1], :])**2)) #np.linalg.norm(im[p1[0], p1[1], :] - im[p2[0], p2[1], :])
    #ret /= 255**2
    #ret = 1000
    return ret

def neighbor_grad_assign(im, point, curr_assign):
    grad_assign = []
    if point[0] - 1 >= 0: 
        point2 = (point[0] - 1, point[1])
        ga = (calc_pix_grad(im, point, point2), curr_assign[point2[0], point2[1]])
        grad_assign.append(ga)
    if point[0] + 1 < im.shape[0]:
        point2 = (point[0] + 1, point[1])
        ga = (calc_pix_grad(im, point, point2), curr_assign[point2[0], point2[1]])
        grad_assign.append(ga)
    if point[1] + 1 < im.shape[1]:
        point2 = (point[0], point[1] + 1)
        ga = (calc_pix_grad(im, point, point2), curr_assign[point2[0], point2[1]])
        grad_assign.append(ga)
    if point[1] - 1 >= 0:
        point2 = (point[0], point[1] - 1)
        ga = (calc_pix_grad(im, point, point2), curr_assign[point2[0], point2[1]])
        grad_assign.append(ga)
    return grad_assign

def perform_sa_cy(im, crf_params, loc_probs, anneal_sched, plot_every=True, live_plot=False):
    d, eta0, alpha, t = crf_params
    seg_mat, superpix_probs = loc_probs
    pix_prb_new = [None for _ in xrange(len(superpix_probs))]
    for key in superpix_probs:
        pix_prb_new[key] = superpix_probs[key]/np.sum(superpix_probs[key])
    pix_prb_new = np.array(pix_prb_new)
    
    assign = sa_cy.perform_sa(
                im.astype(np.float64), 
                seg_mat.astype(np.float64), 
                pix_prb_new,
                anneal_sched,
                plot_every,
                live_plot,
                d,
                eta0,
                alpha,
                t)

    plt.imshow(assign, cmap=new_map)
    plt.show()
    return assign
    
def perform_sa(im, crf_params, loc_probs, anneal_sched, init_assign=None, plot_every=1, live_plot=False):
    """
    im: image in RGB
    crf_params: some params in silberman's paper
    loc_probs: the class probabilities at each pixel location, will be a tuple of superpixel assignment and probability for that assignment
    anneal_sched: annealing schedule
    init_assign: the initial assignment - if None, we sample from loc_probs
    """
    d, eta0, alpha, t = crf_params
    seg_mat, superpix_probs = loc_probs
    H = im.shape[0]
    W = im.shape[1]
    num_classes = superpix_probs[0].shape[0]
    curr_probs = np.zeros((num_classes,))
    rgb_im = skimage.color.rgb2gray(im)
 
    if init_assign is None: 
        init_assign = np.zeros((H, W))   
        for h in xrange(H): 
            for w in xrange(W):
                prbs = superpix_probs[seg_mat[h, w]]
                init_assign[h, w] = np.random.choice(prbs)
    
    curr_assign = init_assign   

    for ind, temp in enumerate(anneal_sched):
        for h in xrange(H):
            for w in xrange(W):
                curr_loc_probs = superpix_probs[seg_mat[h, w]]
                grad_assign = neighbor_grad_assign(rgb_im, (h, w), curr_assign)
                curr_probs = -np.log(curr_loc_probs)
                for ga in grad_assign: 
                    curr_probs += d*eta0*np.exp(-alpha*max(ga[0] - t, 0))
                    curr_probs[ga[1]] -= d*eta0*np.exp(-alpha*max(ga[0] - t, 0))
                curr_probs *= -1.0/temp
                curr_probs -= np.amax(curr_probs)
                curr_probs = np.exp(curr_probs)
                curr_probs /= curr_probs.sum()
                curr_assign[h, w] = np.random.choice(num_classes, p=curr_probs)
        if live_plot and (ind % plot_every == 0):
            plt.imshow(curr_assign, cmap=new_map)            
            plt.show()

    for h in xrange(H):
        for w in xrange(W):
            curr_loc_probs = superpix_probs[seg_mat[h, w]]
            grad_assign = neighbor_grad_assign(rgb_im, (h, w), curr_assign)
            curr_probs = -np.log(curr_loc_probs)
            for ga in grad_assign:
                curr_probs += d*eta0*np.exp(-alpha*max(ga[0] - t, 0))
                curr_probs[ga[1]] -= d*eta0*np.exp(-alpha*max(ga[0] - t, 0))
            curr_assign[h, w] = np.argmax(-curr_probs)

    plt.imshow(curr_assign, cmap=new_map)
    plt.show()
    return curr_assign        

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
    assign = perform_sa_cy(im, (d, eta0, alpha, t), (seg_mask, pix_probs), np.linspace(1, 0.01, 10))
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

def plot_pix_grads(imfile):
    im = skimage.io.imread(imfile)
    #im = skimage.color.rgb2gray(im)
    H = im.shape[0]
    W = im.shape[1]
    dummy_assign = np.zeros((H, W))
    grad_map = np.zeros((H, W))
    for h in xrange(H):
        for w in xrange(W):
            grad_assign = neighbor_grad_assign(im, (h, w), dummy_assign)
            grad_mean = np.mean(np.array([ga[0] for ga in grad_assign]))
            grad_map[h, w] = grad_mean
    normalized_map = grad_map/np.amax(grad_map)*255
    plt.imshow(normalized_map, cmap='gray')
    plt.show()    

def main():
    global args
    parser = ap.ArgumentParser()
    parser.add_argument('--SUN_dir', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--feat_dir', type=str, help='The directory where all the HOG features are stored.') 
    parser.add_argument('--network_path', type=str, help='Path of the neural network we use.')
    parser.add_argument('--imfile', type=int, help='Which image we decide to use.')
    parser.add_argument('--mode', type=str, default='SEGMENT', help='What function to call.')
    parser.add_argument('--d', type=float, default=3, help='Parameter in the crf, see NYU paper 1.')
    parser.add_argument('--eta0', type=float, default=10, help='See first NYU paper for parameter description.')
    parser.add_argument('--alpha', type=float, default=4, help='See first NYU paper for parameter description.')
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
