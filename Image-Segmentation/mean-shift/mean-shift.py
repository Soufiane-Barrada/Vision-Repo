import time
import os
import random
import math
import torch
import numpy as np
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    return torch.sum((X - x) ** 2, dim=1)

def distance_batch(x, X):
    return torch.cdist(x,X) ** 2

def gaussian(dist, bandwidth):
    amplitude = 1 / ((2 * math.pi) ** (3 / 2))
    return amplitude * torch.exp(-dist / (2 * bandwidth ** 2))

def update_point(weight, X):
    #print(weight.shape)
    #print(weight.unsqueeze(1).shape)
    #print(X.shape)
    #print((weight.unsqueeze(1) * X).shape)
    return torch.sum(weight.unsqueeze(1) * X, dim=0)/torch.sum(weight)

def update_point_batch(weight, X):
    #print(weight.shape)
    #print(X.shape)
    #print(torch.mm(weight,X).shape)
    #print(torch.sum(weight, dim=1, keepdim=True).shape)
    return torch.mm(weight,X)/ torch.sum(weight, dim=1, keepdim=True)

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ =X.clone()
    dist= distance_batch(X,X)
    weight= gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
#X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy() 
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
