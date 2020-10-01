#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torchvision as thv
from torchvision import transforms
import  torch as th
from torch.utils.data import DataLoader
#from utils import  check_mkdir, data_split, imshow, minibatch_data, CNN_loading
#from utils import  train_epoch, data_iter, transfer, projection, pw0
#from utils import  test_target, test_source, test, embedding
from utils import *
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
    #transforms.Resize( 224, interpolation=2),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])

transform_test = transforms.Compose([
    #transforms.Scale(224), 
    #transforms.Resize( 224, interpolation=2),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])

########################
stat = defaultdict(dict)
id_list = defaultdict(dict)
####sample size, vl size, dl size
stat['mb_ratio'] = 100 
stat['vl_ratio'] = 4
stat['dl_ratio'] = 15
#stat[ 'n_epochs' ] = 350
stat[ 'n_epochs' ] = 300
stat['p_epoch'] = 150
#stat['T'] = int(( stat['train_size'] / stat['bsize']) * stat[ 'n_epochs' ]) 
stat['T'] = int( stat['dl_ratio'] * stat[ 'n_epochs' ] )
stat['num_period'] = 50 
stat['interval'] = int( stat['T'] / stat['num_period']  )

stat['weight_decay'] = 1e-3
stat['momentum'] = 0.0
stat['iterations'] = 6 #num for itrs for couplings 
stat['la'][0] = 0
stat['num_minibatch'] = 8
#stat['proxy'] = 1.0
stat['proxy'] = 0.0
stat['ot_proxy'] = 0.0

#saving path
MNIST_tran_ini = './CIFAR_initialstatus'
stat['savingmodel'] = './CIFAR_stat'
check_mkdir(stat['savingmodel'] )
check_mkdir(MNIST_tran_ini )
cifar10 = [ 'animal', 'vehicle' ]
cifar100 = [ 'flowers', 'mammals-1', 'mammals-2', 'vehicles-1', 'vehicles-2' ]

id_list['animal'] = [ 2, 3, 4, 5, 6, 7 ]
id_list['vehicle']  = [ 0, 1, 8, 9 ]
id_list['mammals-1'] = [15, 19, 21, 31, 38]
id_list['mammals-2'] = [3, 42, 43, 88, 97]
id_list['vehicles-1'] = [8, 13, 48, 58, 90]
id_list['vehicles-2'] =[ 41, 69, 81, 85, 89]
id_list['flowers'] = [ 54, 70, 62, 82, 92 ]

###define source and target task, gpu
stat['task1'], stat['task2'] = 'vehicles-2', 'vehicles-1' 
stat['dev'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu' 


# In[ ]:





# In[2]:


###########
index1, index2 = id_list[stat['task1']], id_list[stat['task2']]
stat['num_classes'] = int(len(index1) + len(index2))
task = [stat['task1'],  stat['task2']]
##source and target dataset
for i in range (2):
    if i == 0:
        index = index1
    else:
        index = index2
    if task[i] in cifar10:
        train = thv.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)  
        val = thv.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_test)
    else:
        train = thv.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train) 
        val = thv.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_test) 
    #train = data_split(train, index)
    shift_list = [ 0, len(index1) ]
    train = data_split(train, index, shift_list[i])
    val = data_split(val, index, shift_list[i])
    if i == 0:
        source_train, source_val = train, val
    else:
        target_train, target_val = train, val

if stat['task2'] in cifar10:
    test_target_data = thv.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)
else:
    test_target_data = thv.datasets.CIFAR100(root='./data', train=False,
                                         download=True, transform=transform_test)
    
test_target_data = data_split(test_target_data, index2, len(index1))
tl = DataLoader( test_target_data, batch_size=1000, 
                             shuffle=False, drop_last = False)


# In[3]:


len( source_train ), len(target_train), len( source_val ), len(target_val)


# In[4]:


minibatch_data( stat, source_train, source_val, target_train, target_val )


# In[5]:


######saving experiments
stat['p'] = defaultdict(dict)
stat['traj'] = defaultdict(dict)
save = './checkpoint'
check_mkdir(save)


# In[ ]:


for mb in range( stat['num_minibatch'] ):
    stat['proxy'] = 1/ ( 2**mb )
    print(mb, 'we are now in this minibatch')
    stat[ 'distance' ][mb] = defaultdict(dict)
    ####sampling minibatch data
    minibatch_data( stat, source_train, source_val, target_train, target_val )
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
        #stat['norm'][itr] = 0.0
        ####loading model block
        network, optimizer = CNN_loading( stat, MNIST_tran_ini)
        projection(network, optimizer, tl, MNIST_tran_ini, stat, saving, mb, itr)
        stat[ 'distance' ][ mb ][ itr ] = torch.tensor( stat['cp'][itr] * stat['r_dist'][itr] ).sum()
        saving['distance'][mb][itr]= stat[ 'distance' ][mb][itr]
        ############
        if itr == 0:
            copy_traject( stat, mb )
            stat['rie_dist'][mb] = stat[ 'distance' ][mb][itr] + 0. 
        elif stat[ 'distance' ][mb][itr]<= stat['rie_dist'][mb]:
            copy_traject( stat, mb )    
            stat['rie_dist'][mb] = stat[ 'distance' ][mb][itr]+ 0.
            print('updating trajectory')
        ###########
        #stat['norm'][itr] /= stat['T']
        #print(stat['norm'], 'trajectory differences#######')
        print( stat[ 'distance' ][mb], 'riemann distance at ', itr )
        print( torch.tensor( stat['cp'][itr + 1] * stat['r_dist'][itr] ).sum() )
    print(stat['rie_dist'], 'minibatch rieman distance')
        #print( torch.tensor( stat['cp'][itr + 1] * stat['tr_loss'][itr] ).sum(), 'loss' )


# In[ ]:


saving['rie_dist'] = stat['rie_dist']


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





# In[6]:


# pre train model on source task and save the model
###initial dataloader
index = index1
if stat['task1'] in cifar10:
    train = thv.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform_train)
    test_data = thv.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform_test)
else:
    train = thv.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)   
    test_data = thv.datasets.CIFAR100(root='./data', train=False,
                                     download=True, transform=transform_test)
#train = thv.datasets.CIFAR10(root='./data', train=True,
#                                     download=True, transform=transform_train)  

train = data_split(train, index1, 0)
dl = DataLoader( train, batch_size=20, 
                             shuffle=True, drop_last = False)
vl = DataLoader( train, batch_size=100, 
                             shuffle=False, drop_last = False)


test_data = data_split(test_data, index1, 0)
test_loader = DataLoader( test_data, batch_size=100, 
                             shuffle=False, drop_last = False)


# In[ ]:





# In[ ]:





# In[7]:


network = CNN(stat['num_classes']).to(stat['dev'])
optimizer = optim.SGD( network.parameters()
                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay']
)


# In[8]:


for epoch in range( 120 ):
    train_epoch(network, stat, optimizer, dl)
    if (epoch + 1) %30 == 0:
        test(stat, network, vl )
        print('######test results#####')
        test(stat, network, test_loader )


# In[9]:


####saving pretrained model
torch.save(
    network.state_dict(), 
                   os.path.join(MNIST_tran_ini, 
                               'CNN={}.pth'.format( (stat['task1'], stat['task2'] 
                               ) )  
                               )
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


network = CNN(stat['num_classes']).to(stat['dev'])
network.load_state_dict(
    torch.load(
        os.path.join(
            MNIST_tran_ini, 'CNN={}.pth'.format( (stat['task1'], stat['task2']) )
        )))
test(stat, network, vl )
test(stat, network, test_loader )


# In[ ]:




