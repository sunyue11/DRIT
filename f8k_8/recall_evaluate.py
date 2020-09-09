###################双向检索评估效果######################################
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy 
import sys
import importlib
importlib.reload(sys)
import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os
import random
def rank_i2t(images, captions,npts=None):

    if npts is None:
        npts = int(images.shape[0] / 5)

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
		
    length=captions.shape[0]
    for index in range(npts):
    
        # Get query image
        im = images[5 * index]
        distance={}
        for i in range(length):
            distance[i]= np.linalg.norm(im - captions[i])
        distance_sorted = sorted(distance.items(), key=lambda x:x[1])
    
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(np.array(distance_sorted) == distance[i])[0][0]
            if tmp < rank:
                rank = tmp#选择五个中离他最近的index
        ranks[index] = rank
		
    
    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10), medr

def rank_t2i( images, captions,npts=None):

    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for i in range(len(captions)):
        cap = captions[i]
        distance={}
        for index in range(npts):
            distance[index] = np.linalg.norm(cap - ims[index])
        distance_sorted = sorted(distance.items(), key=lambda x:x[1])
		
        tmp = np.where(np.array(distance_sorted) == distance[int(i/5)])[0][0]
        ranks[i] = tmp
            

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10), medr

	
	
