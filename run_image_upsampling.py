"""
Created on June 08, 2021
@author: Zhongjie Yu
@author: Mingye Zhu
Script to do image upsampling on './data/baboon.png'
"""
from PIL import Image
import numpy as np
from MOMoGPstructure import query, build_MOMoGP
from MOMoGP import structure
import torch
from utils import calc_rmse
import random
import torch

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
torch.cuda.manual_seed(23)

# Load original "baboon.png" image, shape (H, W, 3)
image_name = 'baboon.png'
img_original = np.array(Image.open('./data/' + image_name))
H,W,_ = img_original.shape
print("Loaded image with Hight", H, "and Width", W)
scale = 2

# Downsample the original image to shape(H/scale, W/scale, 3)
img_train = img_original[::scale, ::scale, :] 
print("Image size after down sampling:", img_train.shape)
x_train = []
y_train = []
for i in range(int(H/scale)):
    for j in range(int(W/scale)):
        x_train.append([i,j])
        y_train.append(img_train[i,j])

# Set pixel value to [0,1]
x_train = np.asarray(x_train)/H
y_train = np.asarray(y_train)/255
D = x_train.shape[1]
P = y_train.shape[1]

# Set ground truth and index for upsampling
y_test = img_original.reshape((H*W, 1, 3))/255
# index for the complete image
# for simplicity, we apply regression on all pixels, 
# and then assign original pixel values back to their locations
x_test = []
for i in range(H):
    for j in range(W):
        x_test.append([i/scale,j/scale])
x_test = np.asarray(x_test)/H

# Set hyperparameters
Kpx = 2
M = 300
lr = 0.1
epoch = 75
cuda = True

# Args for structure learning
opts = {
        'min_samples': 0,
        'X': x_train,
        'Y': y_train,
        'qd': Kpx-1,
        'max_depth': 100,
        'max_samples': M,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }

# Built the root structure 
root_region, gps_ = build_MOMoGP(**opts)
root, gps = structure(root_region,scope = [i for i in range(P)], gp_types=['matern1.5_ard'])

# Train GP experts with their own hyperparameters
for i, gp in enumerate(gps):
    idx = query(x_train, gp.mins, gp.maxs)
    gp.x = x_train[idx]
    y_scope = y_train[:,gp.scope]
    gp.y = y_scope[idx]
    print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
    gp.init(cuda=cuda, lr=lr, steps=epoch, iter=i)
root.update()

# Upsampling with index from x_test
mu_test,_ = root.forward(x_test, y_d=P)
mu_test = mu_test.reshape((H,W,3))
# Replace with original pixels
for i in range(int(H/scale)):
    for j in range(int(W/scale)):
        mu_test[i*scale,j*scale,:] = img_train[i,j,:]/255
# Map to [0,255], and save the image
mu_test[mu_test<0]=0
mu_test[mu_test>1]=1
mu_test = mu_test*255
mu_test = mu_test.astype(np.uint8).reshape((H,W,3))
im = Image.fromarray(mu_test)
im.save('MOMoGP_upsampling.png')
print("Image saved.")




