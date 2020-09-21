#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, pdb, sys, json, subprocess,        time, logging, argparse,        pickle, math, gzip, numpy as np,        glob
       #pandas as pd, 
       
from functools import partial, reduce
from pprint import pprint
from copy import deepcopy

import  torch as th, torch.nn as nn,         torch.backends.cudnn as cudnn,         torchvision as thv,         torch.nn.functional as F,         torch.optim as optim

from torchvision.datasets import MNIST as mnist
from torchvision import transforms
cudnn.benchmark = True
import torch

from collections import defaultdict
from torch._six import container_abcs

import torch
from copy import deepcopy
from itertools import chain
from model import ResNet18, ResNet34, ResNet50

import math
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import random

# save model
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
#taget dataset validation
def test_target(stat, network, itr):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in stat['tvl']:
            data, target = data.to(stat['dev']), target.to(stat['dev'])
            #output = network(data)
            output = F.log_softmax(  network(data) )
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(stat['tvl'].dataset)
    stat['target_accu'][itr].append( 100. * correct / len(stat['tvl'].dataset) )
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(stat['tvl'].dataset), 
        100. * correct / len(stat['tvl'].dataset)))
           
#source dataset validation
def test_source(stat, network, itr):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in stat['svl']:
            data, target = data.to(stat['dev']), target.to(stat['dev'])
            #output = network(data)
            output = F.log_softmax(  network(data) )
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(stat['svl'].dataset)
    stat['source_accu'][itr].append( 100. * correct / len(stat['svl'].dataset) )
    #test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(stat['svl'].dataset), 
        100. * correct / len(stat['svl'].dataset)))

# train 
def train_epoch(network, stat, optimizer):
    network.train()
    for batch_idx, (data, target) in enumerate( stat['sdl'] ):
        optimizer.zero_grad()
        data, target = data.to(stat['dev']), target.to(stat['dev'])
        output = F.log_softmax(  network(data) )
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
            
##test
def test(stat, network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in stat['svl']:
            data, target = data.to(stat['dev']), target.to(stat['dev'])
            output = F.log_softmax(  network(data) )
            test_loss += float( F.nll_loss(output, target, size_average=False).item() )
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(stat['svl'].dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(stat['svl'].dataset), 
        100. * correct / len(stat['svl'].dataset)) 
         )
 
### self-defined data loader based on the couplings between source and target
def data_iter( stat, cp):
    ns, nt = len( stat['source'] ), len( stat['target'] )
    sp = ns * cp   
    batch_size = stat['bsize']
    num_examples = ns
    indices = list(range(num_examples)) 
    random.shuffle(indices)         
    for i in range(0, num_examples, batch_size):
        j = list( indices[i: min(i + batch_size, num_examples)] )
        batch = torch.utils.data.Subset(stat['source'], j)
        for k in range( batch_size  ):
            xs, ys = batch[k]    
            xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
            
            # random sampling target images based on couplings
            nk = int(np.random.choice(a=nt, size=1, replace=False, p=sp[ j[k] ] ))            
            xt, yt = stat['target'][nk]
            xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev']) 
            
            #outputs batch of souce data and corresponding target data
            if k == 0:
                image_s, image_t, label_s, label_t = xs, xt, ys, yt
            else:
                image_s, image_t, label_s, label_t = torch.cat((image_s, xs), 0
                                                              ), torch.cat((image_t, xt), 0
                                                                          ), torch.cat((label_s, ys), 0
                                                                                    ), torch.cat((label_t, yt), 0)
        
        yield image_s, label_s, image_t, label_t
        
        
####
def projection(network, MNIST_tran_ini, stat, saving, itr):
    optimizer = optim.SGD( network.parameters()
                          , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay'] )
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    #cp = stat['cp'][itr]
    stat['source_accu'][itr] = []
    stat['target_accu'][itr] = []
    stat['loss'][itr] = [] 
    stat['r_dist'][itr] = np.zeros( (ns ,  nt)  )
    stat['tr_loss'][itr] = np.zeros( (ns ,  nt)  )
    print(itr,'itr')
    
    ####transfer block#####    
    start = time.time()
    t = 0 
    for epoch in range(stat['n_epochs']):
        t, stat =  transfer(itr, t, network, optimizer, stat, epoch)              
        if (epoch+1) % 10 == 0 or epoch == 0:
            print('#####source loss######')
            test_source(stat, network, itr)
            print('#####target loss######')
            test_target(stat, network, itr)
            print('Time used is ', time.time() - start)
    print('################')
        
    ps = np.ones( (ns,) ) / ns
    pt = np.ones( (nt, ) )/nt
    #if itr >0 :
    #    stat['r_dist'][itr] = 0.5 * stat['r_dist'][itr] + 0.5 * stat['r_dist'][itr-1] + 0.
    
    #cost = stat['r_dist'][itr] + 0. 
    cost = stat['tr_loss'][itr] + 0. 
    cp = stat['cp'][itr] + 0. 
    
    ### solving regularized optimal transportation 
    reg =  2 ** ( itr - 1 )
    def f(G):
        return - ( G *  cp).sum()   
    def df(G):
        return - cp
    stat['cp'][( itr + 1 )] = ot.optim.cg(ps, pt, cost, reg, f, df, G0 = cp, verbose=True)
    
    #stat['cp'][( itr + 1 )] = 0.1 *  ( ot.emd(ps, pt, cost) + 0. ) + 0.9 * ( stat['cp'][itr] + 0.) 
    #stat['cp'][( itr + 1 )] = ot.emd(ps, pt, cost) + 0.
    #copy down the data
    saving['cp'][itr]= stat['cp'][itr]
    saving['loss'][itr]  = stat['loss'][itr] 
    saving['r_dist'] = stat['r_dist']
    print(t, 'T')
    print('Time used is ', time.time() - start)
    print('itr_end' )    
    
###### a transfer learning epoch
def transfer(itr, t, network, optimizer, stat, epoch):  
    criterion = nn.CrossEntropyLoss()
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    
    #couplings at current iteration
    cp = stat['cp'][itr]
    
    #self-defined data loader
    for d in data_iter(stat, cp):   
        
        # time plus one
        t += 1
        #stat['la'][t] = t/ stat['T'] +0. 
        stat['la'][t] = np.random.beta( a = t/ stat['T'], b = (1 - t/ stat['T'] + 1e-8) )
                          
        ####SGD algorithm
        network.train()
        xs, ys, xt, yt = d
        optimizer.zero_grad()
        # forward + backward + optimize
        x = (1 - stat['la'][t]) * xs + stat['la'][t] * xt
        outputs = network( x )
        loss = (1 - stat['la'][t]) * criterion(outputs, ys) + stat['la'][t] * criterion(outputs, yt)
        loss.backward()
        optimizer.step()        
        stat['loss'][itr].append(float(loss ))
        
        ####model prediction for each pair of mixing images
        interval = stat['interval']
        if t % interval == 0:
            network.eval()       
            with torch.no_grad():
                for m in range( ns ): 
                    for n in range( nt ):
                        xs, ys = stat['source'][ m ]
                        xt, yt = stat['target'][ n ]
                        xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
                        xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev'])
                        x = (1 - stat['la'][t]) * xs + stat['la'][t] * xt
                        
                        ###loss###
                        #outputs = network( x )                        
                        #stat['tr_loss'][itr][m][n] += float( (1 - stat['la'][t]) * criterion(outputs, ys
                        #                                                           ) + stat['la'][t] * criterion(outputs, yt))
                        
                        ##### computing p_{ w_t } ( y | x_t ) 
                        kt = int( t/ interval )
                        stat['pred'][ ( kt, m, n) ] = F.softmax(  network(x) )
                        
                        #### computing p_{ w_t } ( y | x_{ t - interval } ) ##
                        x1 = (1 - stat['la'][t - interval]) * xs + stat['la'][t - interval ] * xt
                        pred1 = F.softmax(  network(x1) )
                        
                        #### call p_{ w_{ t - interval } } ( y | x_{ t - interval } )
                        pred2 = stat['pred'][ ( kt - 1, m, n) ] 
                        
                        ####KL divergence KL( p_{ w_{ t - interval } } ( y | x_{ t - interval } ) || p_{ w_t } ( y | x_{ t - interval } ) )
                        kl = F.relu(F.kl_div( pred1.log(), 
                                      pred2, None, None, 'sum'))
                        
                        #integration of riemann distance
                        stat['r_dist'][itr][m][n] += float( ( math.sqrt( kl ) * interval ) / stat['T'] )
    return t, stat

