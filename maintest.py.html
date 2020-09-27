#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torchvision as thv
from torchvision import transforms
import  torch as th
from torch.utils.data import DataLoader
#from model import CNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
#from Res_model import ResNet50, Identity, fcNet, CNN, CNet_torch

import os, pdb, sys, json, subprocess,        time, logging, argparse,        pickle, math, gzip, numpy as np,        glob

from backpack import extend, backpack
from backpack.extensions import BatchGrad, SumGradSquared, Variance, BatchL2Grad
#from gpu_memory_log import gpu_memory_log

#import matplotlib.pylab as pl
import random
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss


# In[2]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])


# In[3]:


##animal and vehicle dataset
for i in range (2):
    if i == 0:
        index = [ 2, 3, 4, 5, 6, 7 ]
    else:
        index = [ 0, 1, 8, 9 ]
    train = thv.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)  
    train.targets = torch.tensor( train.targets )
    for k in range( len(index) ):
        if k == 0:
            idx = train.targets == index[k]
        else:
            idx += train.targets == index[k]
    train.targets= train.targets[idx]
    train.data = train.data[idx.numpy().astype(np.bool)]
    train0 = train
    #train0, _ = torch.utils.data.random_split(train, 
                                                               # [train_size, len(train)- train_size ])
    
    if i == 0:
        animal = train0
    else:
        vehicle = train0


# In[5]:


index = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
train = thv.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train) 
train.targets = torch.tensor( train.targets )
for k in range( len(index) ):
    if k == 0:
        idx = train.targets == index[k]
    else:
        idx += train.targets == index[k]
train.targets= train.targets[idx]
train.data = train.data[idx.numpy().astype(np.bool)]
cifar100 = train
    


# In[ ]:





# In[6]:


from task2vec import Task2Vec
from models import get_model
import datasets
import task_similarity

dataset_names = ('animal', 'vehicle', 'cifar100', 'cifar10' )
#dataset_list = [datasets.__dict__[name]('./data')[0] for name in dataset_names] 
dataset_list =[ animal, vehicle, cifar100, 
               datasets.__dict__['cifar10']('./data')[0] ]
embeddings = []
for name, dataset in zip(dataset_names, dataset_list):
    print(f"Embedding {name}")
    probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets)+1)).cuda()
    embeddings.append( Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset) )
task_similarity.plot_distance_matrix(embeddings, dataset_names)


# In[16]:


embedding =  Task2Vec(probe_network).embed(animal)


# In[10]:


from ipywidgets import IntProgress


# In[ ]:




