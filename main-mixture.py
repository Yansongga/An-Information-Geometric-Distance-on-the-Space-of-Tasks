#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torchvision as thv
from torchvision import transforms
import  torch as th
from torch.utils.data import DataLoader
from utils import  check_mkdir, data_split, imshow, minibatch_data, CNN_loading
from utils import  train_epoch, data_iter, transfer, projection, pw0, minibatch_data_mix
from utils import  test_target, test_source, test, embedding, projection_mix, transfer_mix, pw0_mix
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
#stat['train_size'] = 100
stat[ 'n_epochs' ] = 4
#stat['bsize'] = 4
#stat['dsize'] = 50
stat['weight_decay'] = 1e-3
#stat['iterations'] = 8 #num for itrs for couplings
stat['T'] = int( 1000 * stat[ 'n_epochs' ] ) 
stat['interval'] = int( stat['T'] / 50) 
stat['la'][0] = 0
#stat['num_minibatch'] = 10
#stat['proxy'] = 1.0

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
stat['dev'] = 'cuda: 2' if torch.cuda.is_available() else 'cpu' 


# In[2]:


a = torch.tensor( [[0.3, 0.4, 0.3], [0.2, 0.2, 0.6] ])
b = torch.tensor( [[0.1, 0.2, 0.7], [ 0.1, 0.3, 0.6] ])
( a * ( a.log() - b.log() ) ).sum(1).shape


# In[3]:


0.3* np.log(3) + 0.4 * np.log(2) + 0.3 * np.log(3/7)


# In[4]:


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


# In[5]:


len( source_data ), len(target_data)


# In[6]:


######saving experiments
stat['p'] = defaultdict(dict)
stat['traj'] = defaultdict(dict)
save = './checkpoint'
check_mkdir(save)


# In[7]:


itr = 0 
####sampling minibatch data
minibatch_data_mix( stat, source_data, target_data )
####loading pre-trained model
network, optimizer = CNN_loading( stat, MNIST_tran_ini)
######### p_{w_0}( y| x ) on source task
pw0_mix(stat, network)
#couplings updates block
saving = defaultdict(dict)
####loading model block
network, optimizer = CNN_loading( stat, MNIST_tran_ini)
projection_mix(network, optimizer, MNIST_tran_ini, stat, saving, itr)
############
###########
print( stat[ 'r_dist' ], 'riemann distance at ', itr )
#print( torch.tensor( stat['cp'][itr + 1] * stat['tr_loss'][itr] ).sum(), 'loss' )


# In[6]:


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





# In[5]:


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





# In[6]:


network = CNN().to(stat['dev'])
optimizer = optim.SGD( network.parameters()
                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay']
)


# In[7]:


for epoch in range( 40 ):
    train_epoch(network, stat, optimizer, dl)
    if (epoch + 1) %5 == 0:
        test(stat, network, vl )
        print('######test results#####')
        test(stat, network, test_loader )


# In[8]:


####saving pretrained model
torch.save(
    network.state_dict(), 
                   os.path.join(MNIST_tran_ini, 
                               'CNN={}.pth'.format( stat['task1'] )
                               )
)  


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

