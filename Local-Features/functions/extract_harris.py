import numpy as np
from scipy.signal import convolve2d
import cv2
from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
from scipy.ndimage import maximum_filter

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''

    img = img.astype(float) / 255.0

    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

    Ix= convolve2d(img,sobel_x,mode='same') # maybe boundary= 'symm"
    Iy= convolve2d(img,sobel_y,mode='same') # maybe boundary= 'symm"
    
    
    # Blur the computed gradients
    opt_sig= 1.0
    opt_ker=(5,5)
    Ix_blurred = cv2.GaussianBlur(Ix, opt_ker, opt_sig, borderType=cv2.BORDER_REPLICATE)
    Iy_blurred = cv2.GaussianBlur(Iy, opt_ker, opt_sig, borderType=cv2.BORDER_REPLICATE)

    # Compute elements of the local auto-correlation matrix "M"
    Ix2= Ix_blurred * Ix_blurred
    Iy2= Iy_blurred * Iy_blurred
    Ixy= Ix_blurred * Iy_blurred
    size=(9,9)
    sum_Ixx = cv2.GaussianBlur(Ix2, size, sigma, borderType=cv2.BORDER_REPLICATE)
    sum_Iyy = cv2.GaussianBlur(Iy2, size, sigma, borderType=cv2.BORDER_REPLICATE)
    sum_Ixy = cv2.GaussianBlur(Ixy, size, sigma, borderType=cv2.BORDER_REPLICATE)

    # Compute Harris response function C
    determinant = (sum_Ixx * sum_Iyy) - (sum_Ixy ** 2)
    trace = sum_Ixx + sum_Iyy

    R = determinant - k * (trace ** 2)

    # Detection with threshold and non-maximum suppression
    max_size=(9,9) # (7,7)
    R_max = maximum_filter(R, size=max_size)
    corners = np.where((R == R_max) & (R_max > thresh))
    corners = np.stack((corners[1], corners[0]), axis=-1)

    return corners, R

