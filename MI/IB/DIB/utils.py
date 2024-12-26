from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def pairwise_distnaces(x):
    # Sample 간의 유클리디안 거리
    x = x.view(128, -1) # Batch가 128이라서 128개의 input image (32*32)간의 거리
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    return -2*torch.mm(x, x.t()) + instances_norm + instances_norm.t() # (128, 32*32)

def calculate_gram_mat(x, sigma):
    dist = pairwise_distnaces(x)
    # Gram matrix K_ij = exp(-dist / sigma)
    return torch.exp(-dist / sigma)

def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k /torch.trace(k) # normalize 해주고
    eigv = torch.abs(torch.symeig(k, eigenvectors = True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    # reyi_alpha 방식의 Entropy 계산
    return entropy

def joint_entropy(x,y,s_x,s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y) # Kx * Ky
    k = k / torch.trace(k) # Normalize
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha

    entropy = (1/ (1-alpha)) * torch.log2(torch.sum(eig_pow))

    return entropy

def calculate_MI(x, y, s_x, s_y):
    # I(X;Y) = H(X) + H(Y) - H(X, Y)
    Hx = reyi_entropy(x, s_x)
    Hy = reyi_entropy(y, s_y)
    Hxy = joint_entropy(x,y,s_x,s_y)

    Ixy = Hx + Hy - Hxy
    
    return Ixy