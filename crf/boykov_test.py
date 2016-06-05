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
import boykov_seg
import sklearn.metrics

colors = [(random.random(),random.random(),random.random()) for i in xrange(256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

allowed_classes = {u'bed':0, u'bookshelf':1, u'cabinet':2, u'ceiling':3, u'floor':4, u'picture':5, u'sofa':6, u'table':7, u'television':8, u'wall':9, u'window':10}

def assign_superpix_prob(im, network, descs):
    """
    im: image, assumed to be in grayscale
    network: neural network for appearance classification
    descs: list of tuple of (desc_x, desc_y, sift feature)
    """
    if args.seg_type == 'FELZENSZWALB':
	im = skimage.io.rgb2gray(im)
	seg_mask = skimage.segmentation.felzenszwalb(im, scale=100)
    if args.seg_type == 'QUICKSHIFT':
        seg_mask = skimage.segmentation.quickshift(im)
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
  
    #print probs_arr, grad_vert_T, grad_hor_T 
    lab_assign = gco_python.pygco.cut_simple_vh(costs, pairwise_cost, grad_vert_T, grad_hor_T, algorithm='expansion')
    return lab_assign

def segment_im(imfile, featfile, network, d, eta0, alpha, t):
    feat_f = open(featfile, "r")
    im = skimage.io.imread(imfile)    
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

def find_acc(filenum, network, colors, should_save=False, save_name=None, truth_name=None):
    mat_path = args.SUN_dir + ('SUNRGBD/kv1/NYUdata/NYU%04d/seg.mat' % filenum)
    seg_labels = scipy.io.loadmat(mat_path)['seglabel']
    names = scipy.io.loadmat(mat_path)['names']
    imfile = args.SUN_dir + ('SUNRGBD/kv1/NYUdata/NYU%04d/image/NYU%04d.jpg' % (filenum, filenum))
    featfile = args.feat_dir + ('%04d.txt' % filenum)
    assign = segment_im(imfile, featfile, network, args.d, args.eta0, args.alpha, args.t)
    assign = assign.T
    if should_save:
	plt.imsave(save_name, assign, cmap=colors)
    class_map = allowed_classes.copy() if args.hardcode else pickle.load(open(args.class_map, "r"))
    conv_labels = np.zeros(assign.shape)
    original_len = len(class_map)
    for h in xrange(assign.shape[0]):
        for w in xrange(assign.shape[1]):
            curr_name = names[seg_labels[h, w] - 1][0][0]
            if curr_name not in class_map:
                class_map[curr_name] = len(class_map)
            conv_labels[h, w] = class_map[curr_name]

    numpix = assign.shape[0]*assign.shape[1]
    acc_map = 2*(conv_labels != assign)
    invalid_mask = (conv_labels >= original_len) 
    valid_mask = (conv_labels < original_len)
    acc_map[invalid_mask] = 1

    y_flat = conv_labels[valid_mask].flatten()
    assign_flat = assign[valid_mask].flatten()
    cm = sklearn.metrics.confusion_matrix(y_flat, assign_flat, labels=range(original_len))
    row_sum = cm.sum(axis=1).reshape(cm.shape[0], 1)

    if should_save:
	plt.imsave(truth_name, acc_map.astype(float)/2, cmap='gray') 

    return acc_map, cm, row_sum

def main():
    global args
    parser = ap.ArgumentParser()
    parser.add_argument('--SUN_dir', type=str, help='The directory of where SUNRGBD is stored.')
    parser.add_argument('--feat_dir', type=str, help='The directory where all the HOG features are stored.') 
    parser.add_argument('--network_path', type=str, help='Path of the neural network we use.')
    parser.add_argument('--predict_set', type=str, help='Which set we predict on.')
    parser.add_argument('--num_predict', type=int, default=654, help='Number of images to predict on.')
    parser.add_argument('--d', type=float, default=1, help='Parameter in the crf, see NYU paper 1.')
    parser.add_argument('--eta0', type=float, default=10, help='See first NYU paper for parameter description.')
    parser.add_argument('--alpha', type=float, default=0.02, help='See first NYU paper for parameter description.')
    parser.add_argument('--t', type=float, default=0, help='See first NYU paper for parameter description.')
    parser.add_argument('--class_map', type=str, default='../classify/classmap.pkl', help='Class map.')
    parser.add_argument('--hardcode', type=bool, default=False, help='Whether to use hardcoded class map.')
    parser.add_argument('--save_inds', type=str, default=None, help='The indices of the image segmentations to save. If this is none, save in new_save_inds.')
    parser.add_argument('--new_save_inds', type=str, help='Where to save the new randomly generated indies to save.')
    parser.add_argument('--color_map', type=str, default=None, help='Where to find the color map.')
    parser.add_argument('--new_color_map', type=str, help='Where to save color map if none exists right now.')
    parser.add_argument('--im_out', type=str, help='Directory for where to output the saved random images.')
    parser.add_argument('--seg_type', type=str, default='QUICKSHIFT', help='Superpixel segmentation type to use.')
    args = parser.parse_args()
    save_inds = pickle.load(open(args.save_inds, "r")) if args.save_inds is not None else (np.random.random((args.num_predict,)) < 0.1)
    color_map = new_map if args.color_map is None else pickle.load(open(args.color_map, "r"))
    if args.new_save_inds is not None:
	pickle.dump(save_inds, open(args.new_save_inds, "w"))
    if args.new_color_map is not None:
	pickle.dump(color_map, open(args.new_color_map, "w"))
    train_test = scipy.io.loadmat(args.SUN_dir + "splits.mat")
    split = train_test[args.predict_set + 'Ndxs']
    network = pickle.load(open(args.network_path, "r"))
    num_classes = 11 if args.hardcode else len(pickle.load(open(args.class_map, "r")))
    frequencies_tot = np.zeros((num_classes, 1))
    cm = np.zeros((num_classes, num_classes))
    running_total = 0
    curr_acc = float(0)
    for i in xrange(args.num_predict):
        filenum = split[i]
	save_name = args.im_out + ("seg%d.jpg" % filenum)
	truth_name = args.im_out + ("truth%d.jpg" % filenum)
        acc_map, conf_map, frequencies = find_acc(filenum, network, color_map, should_save=save_inds[i], save_name=save_name, truth_name=truth_name)
        total_pix = frequencies.sum()
        curr_acc = (curr_acc*running_total + np.sum(acc_map == 0))/(running_total + total_pix)
	running_total += total_pix
	frequencies_tot += frequencies
	cm += conf_map
    cm = cm.astype(float)/frequencies_tot
    plt.matshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()
    print "Final frequencies", frequencies_tot
    print "Pixel accuracy", curr_acc
 
if __name__ == '__main__':
    main()

