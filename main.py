#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torchvision as thv
from torchvision import transforms
import  torch as th
from torch.utils.data import DataLoader

from utils import  check_mkdir, train_epoch, test_target, test_source, test, data_iter, transfer, projection
#from utils import  train_epoch, data_iter, transfer, projection
#from utils import  test_target, test_source, test, transfer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from model import CNN_torch, CNN
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


batch_size_test = 1000
learning_rate = 1e-3
momentum = 0
log_interval = 12
train_size = 200
stat = defaultdict(dict)
stat[ 'n_epochs' ] = 80
stat['bsize'] = 4
stat['iterations'] = 8 #num for itrs for couplings updates
stat['weight_decay'] = 5e-4
stat['dev'] = 'cuda: 2' if torch.cuda.is_available() else 'cpu' 


# In[4]:


#saving path
MNIST_tran_ini = './CIFAR_initialstatus'
stat['savingmodel'] = './CIFAR_stat'
check_mkdir(stat['savingmodel'] )
check_mkdir(MNIST_tran_ini )


# In[5]:


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
    #train0 = train
    train0, _ = torch.utils.data.random_split(train, 
                                                                [train_size, len(train)- train_size ])
    
    if i == 0:
        stat['source'] = train0
    else:
        stat['target'] = train0


# In[6]:


# define train loader and validation loader
stat['svl'] = DataLoader( stat['source'], batch_size=1000, 
                             shuffle=False, drop_last = False)
stat['tvl'] = DataLoader( stat['target'], batch_size=1000, 
                             shuffle=False, drop_last = False)

stat['sdl'] = DataLoader( stat['source'], batch_size=stat['bsize'], 
                             shuffle=True, drop_last = False)
stat['tdl'] = DataLoader( stat['target'], batch_size=stat['bsize'], 
                             shuffle=True, drop_last = False)


# In[7]:


##display images###
import matplotlib.pyplot as plt 
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
images, labels = stat['source'][1]
#images, labels = sub[0]

# show images
imshow(thv.utils.make_grid(images))


# In[ ]:
####embedding cosine similarities
start = time.time()
ns, nt = len( stat['source'] ), len( stat['target'] )   
stat['embedding'] = np.zeros( (ns ,  nt)  )
probe.eval()
with torch.no_grad():
    for m in range( ns ): 
        print('Time used is ', time.time() - start)
        
        for n in range( nt ):
            for k in range( 8 ):
                xs, ys = stat['source'][ m ]
                xt, yt = stat['target'][ n ]
                xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
                xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev'])
                es, et = probe( xs ), probe( xt )
                #prod = ( es * et ).sum()       
                prod = - torch.cosine_similarity(es, et, dim=1)
                stat['embedding'][m][n] += float( prod ) 
            stat['embedding'][m][n] /= 8
        print( prod )

###compute warm up initial guess coupling
def f(G):
    return np.sum(G * np.log(G))
def df(G):
    return np.log(G) + 1.
reg = 1e-3

ps = np.ones( (ns,) ) / ns
pt = np.ones( (nt, ) )/nt
cost = stat['embedding'] + 0.
#stat['cp'][0] = ot.emd(ps, pt, stat['embedding'] ) + 0. 
stat['cp'][0 ] = ot.optim.cg(ps, pt, cost, reg, f, df, verbose=True)


# In[8]:


# pre train model on source task and save the model
for epoch in range( 20 ):
    train_epoch(network, stat, optimizer)
    if (epoch + 1) %2 == 0:
        test(stat, network )


# In[9]:


torch.save(
    network.state_dict(), 
                   os.path.join(MNIST_tran_ini, 
                               'CNN={}.pth'.format( ( 'animal', 'vehicle' ) )
                               )
)  


# In[ ]:





# In[10]:


# load pre-trained model
network = CNN().to(stat['dev'])
network.load_state_dict(
    torch.load(
        os.path.join(
            MNIST_tran_ini, 'CNN={}.pth'.format( ( 'animal', 'vehicle') )
        )))

optimizer = optim.SGD( network.parameters()
                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay']
)


# In[11]:


#Set up
stat['T'] = int(( len(stat['source']) / stat['bsize']) * stat[ 'n_epochs' ]) 
stat['interval'] = int( stat['T'] / 25) 
stat['la'][0] = 0

###computing model predictions for source images
start = time.time()
network.eval()  
ns, nt = len( stat['source'] ), len( stat['target'] )  
with torch.no_grad():
    for m in range( ns ): 
        for n in range( nt ):
            xs, ys = stat['source'][ m ]
            xt, yt = stat['target'][ n ]
            xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
            xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev'])
            x =  xs
            stat['pred'][ ( 0, m, n) ] = F.softmax(  network(x) )
            
print('Time used is ', time.time() - start)


# In[ ]:


#solving the optimal couplings
saving = defaultdict(dict)
for itr in range( stat['iterations'] ):
    network = CNN()
    network.load_state_dict(
        torch.load(
            os.path.join(
                MNIST_tran_ini, 'CNN={}.pth'.format( ( 'animal', 'vehicle') )
            )))
    network = network.to(stat['dev'])
    projection(network, MNIST_tran_ini, stat, saving, itr)
    stat[ 'distance' ][ itr ] = torch.tensor( stat['cp'][itr] * stat['r_dist'][itr] ).sum()
    saving['distance'][itr] = stat[ 'distance' ][ itr ]
    print( stat[ 'distance' ][ itr ], 'riemann distance at ', itr )
    print( torch.tensor( stat['cp'][itr + 1] * stat['r_dist'][itr] ).sum() )
    print( torch.tensor( stat['cp'][itr + 1] * stat['tr_loss'][itr] ).sum(), 'loss' )


# In[ ]:






