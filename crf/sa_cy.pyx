# cython: profile=True
import numpy as np
import skimage.color

from cpython cimport bool
cimport numpy as np
import cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def calc_pix_grad(
        np.ndarray[DTYPE_t, ndim=3] im, 
        int p1y,
        int p1x, 
        int p2y,
        int p2x):
    """
    Not really sure how to do this so we'll leave it at this for now.
    im: image in RGB
    p1, p2: tuples for the (y, x) of the image coords
    """
    #cdef DTYPE_t ret = np.abs(im[p1y, p1x] - im[p2y, p2x])
    cdef DTYPE_t ret = np.exp(-0.001*np.linalg.norm(im[p1y, p1x, :] - im[p2y, p2x, :])**2)
    #ret /= 255**2
    #ret = 1000
    return ret

def neighbor_grad_assign(
        np.ndarray[DTYPE_t, ndim=3] im, 
        int pointy,
        int pointx, 
        np.ndarray[DTYPE_t, ndim=2] curr_assign
        ):
    cdef int num_neighbors = 0
    cdef np.ndarray neighbor_assign = np.empty(4, dtype=np.int)
    cdef np.ndarray pix_grad = np.empty(4, dtype=DTYPE)
    cdef int point2y, point2x
    if pointy - 1 >= 0:
        point2y = pointy - 1
        point2x = pointx
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = calc_pix_grad(im, pointy, pointx, point2y, point2x), curr_assign[point2y, point2x]
        num_neighbors += 1
    if pointy + 1 < im.shape[0]:
        point2y = pointy + 1
        point2x = pointx
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = calc_pix_grad(im, pointy
, pointx, point2y, point2x), curr_assign[point2y, point2x]
        num_neighbors += 1
    if pointx + 1 < im.shape[1]:
        point2y = pointy
        point2x = pointx + 1
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = calc_pix_grad(im, pointy
, pointx, point2y, point2x), curr_assign[point2y, point2x]
        num_neighbors += 1
    if pointx - 1 >= 0:
        point2y = pointy
        point2x = pointx - 1
        pix_grad[num_neighbors], neighbor_assign[num_neighbors] = calc_pix_grad(im, pointy
, pointx, point2y, point2x), curr_assign[point2y, point2x]
        num_neighbors += 1
    return neighbor_assign, pix_grad, num_neighbors

def perform_sa(
        np.ndarray[DTYPE_t, ndim=3] im,
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
                neighbor_assign, pix_grad, num_neighbors = neighbor_grad_assign(gray_im, h, w, curr_assign)
                curr_probs = -np.log(curr_loc_probs)
                for neighbor_it in xrange(num_neighbors): 
                    add_val = d*eta0*pix_grad[neighbor_it]#np.exp(-alpha*np.max(pix_grad[neighbor_it] - t, 0))
                    curr_probs += add_val
                    #print(add_val)
                    curr_probs[neighbor_assign[neighbor_it]] -= add_val
                curr_assign[h, w] = np.argmax(-curr_probs)
                #curr_probs *= -1.0/temp
                #curr_probs -= np.amax(curr_probs)
                #curr_probs = np.exp(curr_probs)
                #curr_probs /= curr_probs.sum()
                #curr_assign[h, w] = np.random.multinomial(1, curr_probs).argmax()
        #if live_plot and (ind % plot_every == 0):
        #    plt.imshow(curr_assign, cmap=new_map)            
        #    plt.show()

    for w in xrange(W):
        for h in xrange(H):
            curr_loc_probs = superpix_probs[seg_mat[h, w]]
            neighbor_assign, pix_grad, num_neighbors = neighbor_grad_assign(gray_im, h, w, curr_assign)
            curr_probs = -np.log(curr_loc_probs)
            for neighbor_it in xrange(num_neighbors):
                add_val = d*eta0*pix_grad[neighbor_it]#np.exp(-alpha*np.max(pix_grad[neighbor_it] - t, 0))
                curr_probs += add_val
                curr_probs[neighbor_assign[neighbor_it]] -= add_val
            curr_assign[h, w] = np.argmax(-curr_probs)
    ##plt.imshow(curr_assign, cmap=new_map)
    #plt.show()
    return curr_assign    
