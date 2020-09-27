#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torchvision as thv
from torchvision import transforms
import  torch as th
from torch.utils.data import DataLoader
from utils import  check_mkdir, data_split, imshow, minibatch_data, CNN_loading
from utils import  train_epoch, data_iter, transfer, projection, pw0
from utils import  test_target, test_source, test, embedding
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from model import CNN_torch, CNN, Net
import os, pdb, sys, json, subprocess,        time, logging, argparse,        pickle, math, gzip, numpy as np,        glob
from backpack import extend, backpack
from backpack.extensions import BatchGrad, SumGradSquared, Variance, BatchL2Grad
import random
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import matplotlib.pyplot as plt  
 
################
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

########################
stat = defaultdict(dict)
####sample size, vl size, dl size
stat['mb_ratio'] = 100 
stat['vl_ratio'] = 4
stat['dl_ratio'] = 15
stat[ 'n_epochs' ] = 300
stat['p_epoch'] = 60
#stat['T'] = int(( stat['train_size'] / stat['bsize']) * stat[ 'n_epochs' ]) 
stat['T'] = int( stat['dl_ratio'] * stat[ 'n_epochs' ] )
stat['interval'] = int( stat['T'] / 50)

stat['weight_decay'] = 1e-3
stat['momentum'] = 0.0
stat['iterations'] = 5 #num for itrs for couplings 
stat['la'][0] = 0
stat['num_minibatch'] = 10
stat['proxy'] = 1.0
stat['ot_proxy'] = 20.0

#saving path
MNIST_tran_ini = './CIFAR_initialstatus'
stat['savingmodel'] = './CIFAR_stat'
check_mkdir(stat['savingmodel'] )
check_mkdir(MNIST_tran_ini )
animal = [ 2, 3, 4, 5, 6, 7 ]
vehicle = [ 0, 1, 8, 9 ]
cifar10 = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

###define source and target task, gpu
stat['task1'], stat['task2'] = 'animal', 'vehicle'
index1, index2 = animal, vehicle
stat['dev'] = 'cuda: 3' if torch.cuda.is_available() else 'cpu' 

#stat['bsize'] = 10
#stat['dsize'] = 50
#stat['train_size'] = 100


# In[2]:


##source and target dataset
for i in range (2):
    if i == 0:
        index = index1
    else:
        index = index2
    train = thv.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)  
    train = data_split(train, index)
    if i == 0:
        source_data = train
    else:
        target_data = train


# In[3]:


len( source_data ), len(target_data)


# In[4]:


######saving experiments
stat['p'] = defaultdict(dict)
stat['traj'] = defaultdict(dict)
save = './checkpoint'
check_mkdir(save)


# In[ ]:


for mb in range( stat['num_minibatch'] ):
    stat[ 'distance' ][mb] = defaultdict(dict)
    if mb > 0:
        loading = torch.load(
                './checkpoint/trajectory_mb={}.t7'.format( 
                   ( stat['task1'], stat['task2'], mb - 1 )
                )
            )
        stat['proxy_trajectory'] = loading['trajectory']
    ####sampling minibatch data
    minibatch_data( stat, source_data, target_data )
    #####probe network for computing image embeeddings \phi
    embedding(stat)
    ####loading pre-trained model
    network, optimizer = CNN_loading( stat, MNIST_tran_ini)
    ######### p_{w_0}( y| x ) on source task
    pw0(stat, network)
    #couplings updates block
    saving = defaultdict(dict)
    saving['distance'][mb] = defaultdict(dict)
    for itr in range( stat['iterations'] ):
        ####loading model block
        network, optimizer = CNN_loading( stat, MNIST_tran_ini)
        projection(network, optimizer, MNIST_tran_ini, stat, saving, mb, itr)
        stat[ 'distance' ][ mb ][ itr ] = torch.tensor( stat['cp'][itr] * stat['r_dist'][itr] ).sum()
        saving['distance'][mb][itr]= stat[ 'distance' ][mb][itr]
        ############
        if itr == 0:
            tj = { 'trajectory': stat['p']  }
            torch.save( tj, './checkpoint/trajectory_mb={}.t7'.format( 
                ( stat['task1'], stat['task2'], mb ))) 
            stat['rie_dist'][mb] = stat[ 'distance' ][mb][itr] + 0. 
        elif stat[ 'distance' ][mb][itr]<= stat['rie_dist'][mb]:
            stat['rie_dist'][mb] = stat[ 'distance' ][mb][itr]+ 0.
            tj = { 'trajectory': stat['p']  }
            torch.save( tj, './checkpoint/trajectory_mb={}.t7'.format( 
                ( stat['task1'], stat['task2'], mb )))                       
            print('updating trajectory')
        ###########
        print( stat[ 'distance' ][mb], 'riemann distance at ', itr )
        print( torch.tensor( stat['cp'][itr + 1] * stat['r_dist'][itr] ).sum() )
        #print( torch.tensor( stat['cp'][itr + 1] * stat['tr_loss'][itr] ).sum(), 'loss' )


# In[ ]:


print(stat['rie_dist'])


# In[ ]:





# In[ ]:


######saving experiments
save = './checkpoint'
check_mkdir(save)
states = {
    'statistics': saving                   # 将epoch一并保存
}
#torch.save(states, './checkpoint/1st_CIFAR_ymix{}.t7'.format( (7,4) ))
torch.save( states, './checkpoint/CIFAR-task={}.t7'.format( 
        ( stat['task1'], stat['task2'], 'ot' )))


# In[ ]:





# In[ ]:





# In[8]:


# pre train model on source task and save the model
###initial dataloader
index = index1
train = thv.datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)  
train = data_split(train, index)
dl = DataLoader( train, batch_size=20, 
                             shuffle=True, drop_last = False)
vl = DataLoader( train, batch_size=1000, 
                             shuffle=False, drop_last = False)

test_data = thv.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform_train)
test_data = data_split(test_data, index)
test_loader = DataLoader( test_data, batch_size=1000, 
                             shuffle=False, drop_last = False)


# In[ ]:





# In[ ]:





# In[9]:


network = CNN().to(stat['dev'])
optimizer = optim.SGD( network.parameters()
                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay']
)


# In[10]:


for epoch in range( 40 ):
    train_epoch(network, stat, optimizer, dl)
    if (epoch + 1) %5 == 0:
        test(stat, network, vl )
        print('######test results#####')
        test(stat, network, test_loader )


# In[11]:


####saving pretrained model
torch.save(
    network.state_dict(), 
                   os.path.join(MNIST_tran_ini, 
                               'CNN={}.pth'.format( stat['task1'] )
                               )
)  


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


network = CNN().to(stat['dev'])
network.load_state_dict(
    torch.load(
        os.path.join(
            MNIST_tran_ini, 'CNN={}.pth'.format( stat['task1'] )
        )))
test(stat, network, vl )
test(stat, network, test_loader )

