# cython: profile=True
import numpy as np
import skimage.color

from cpython cimport bool
cimport numpy as np
import cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)

def neighbor_grad_assign(
        int pointy,
        int pointx, 
        np.ndarray[DTYPE_t, ndim=2] pix_grad_vert,
	np.ndarray[DTYPE_t, ndim=2] pix_grad_hor,
	np.ndarray[DTYPE_t, ndim=2] curr_assign
        ):
    cdef int num_neighbors = 0
    cdef np.ndarray neighbor_assign = np.empty(4, dtype=np.int)
    cdef np.ndarray pix_grad = np.empty(4, dtype=DTYPE)
    cdef int point2y, point2x
    if pointy - 1 >= 0:
        point2y = pointy - 1
        point2x = pointx
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = pix_grad_vert[point2y, point2x], curr_assign[point2y, point2x]
        num_neighbors += 1
    if pointy + 1 < curr_assign.shape[0]:
        point2y = pointy + 1
        point2x = pointx
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = pix_grad_vert[pointy, pointx], curr_assign[point2y, point2x]
        num_neighbors += 1
    if pointx + 1 < curr_assign.shape[1]:
        point2y = pointy
        point2x = pointx + 1
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = pix_grad_hor[pointy, pointx], curr_assign[point2y, point2x]
        num_neighbors += 1
    if pointx - 1 >= 0:
        point2y = pointy
        point2x = pointx - 1
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = pix_grad_hor[point2y, point2x], curr_assign[point2y, point2x]
        num_neighbors += 1
    return neighbor_assign, pix_grad, num_neighbors

def perform_sa(
        np.ndarray[DTYPE_t, ndim=3] im,
	np.ndarray[DTYPE_t, ndim=2] pix_grad_vert,
	np.ndarray[DTYPE_t, ndim=2] pix_grad_hor,
        np.ndarray[DTYPE_t, ndim=2] seg_mat,
        np.ndarray[DTYPE_t, ndim=2] superpix_probs,
        np.ndarray[DTYPE_t, ndim=1] anneal_sched, 
        bool plot_every, 
        bool live_plot,
        float d,
        float eta0,
        float alpha,
        float t
        ):
    """
    im: image in RGB
    crf_params: some params in silberman's paper
    loc_probs: the class probabilities at each pixel location, will be a tuple of superpixel assignment and probability for that assignment
    anneal_sched: annealing schedule
    init_assign: the initial assignment - if None, we sample from loc_probs
    """
    cdef int H = im.shape[0]
    cdef int W = im.shape[1]
    num_classes = superpix_probs[0].shape[0]
    cdef np.ndarray curr_probs = np.zeros([num_classes,], dtype=np.float64)
    cdef np.ndarray gray_im = im# skimage.color.rgb2gray(im)
 
    cdef int ind, h, w
    cdef DTYPE_t temp
    
    cdef np.ndarray curr_assign = np.zeros([H, W], dtype=np.float64)
    
    for h in xrange(H):
        for w in xrange(W):
            curr_assign[h, w] = np.argmax(superpix_probs[seg_mat[h, w]]) #np.random.multinomial(1, superpix_probs[seg_mat[h, w]]).argmax()
    
    cdef np.ndarray curr_loc_probs    
    cdef int neighbor_it
    cdef DTYPE_t add_val
    
    for ind, temp in enumerate(anneal_sched):
        for w in xrange(W):
            for h in xrange(H):
                curr_loc_probs = superpix_probs[int(seg_mat[h, w])]
                neighbor_assign, pix_grad, num_neighbors = neighbor_grad_assign(h, w, pix_grad_vert, pix_grad_hor, curr_assign)
                curr_probs = -np.log(curr_loc_probs)
                for neighbor_it in xrange(num_neighbors): 
                    add_val = d*eta0*np.exp(-alpha*pix_grad[neighbor_it])
                    curr_probs += add_val
                    curr_probs[neighbor_assign[neighbor_it]] -= add_val
                curr_probs *= -1.0/temp
                curr_probs -= np.amax(curr_probs)
                curr_probs = np.exp(curr_probs)
                curr_probs /= curr_probs.sum()
                curr_assign[h, w] = np.random.multinomial(1, curr_probs).argmax()

    for w in xrange(W):
        for h in xrange(H):
            curr_loc_probs = superpix_probs[seg_mat[h, w]]
            neighbor_assign, pix_grad, num_neighbors = neighbor_grad_assign(h, w, pix_grad_vert, pix_grad_hor, curr_assign)
            curr_probs = -np.log(curr_loc_probs)
            for neighbor_it in xrange(num_neighbors):
                add_val = d*eta0*np.exp(-alpha*pix_grad[neighbor_it])                
                curr_probs += add_val
                curr_probs[neighbor_assign[neighbor_it]] -= add_val
            curr_assign[h, w] = np.argmax(-curr_probs)
    return curr_assign    
