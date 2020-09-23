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
from Res_model import ResNet18, ResNet34, ResNet50

import math

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import random
device = torch.device('cuda: 0')
args = dict(
            dev='cuda: 3' if th.cuda.is_available() else 'cpu',
            bsz=8,
            E=100,
            s=42,
            lr=1e-3,
            wd=1e-4,
            lamda=2,
            gamma=10,
            TT=1,
            alpha=0.01,
            T=10,
            nz=8,
            n_scla = 5,
            n_tcla = 5,
            ns_each = 8,
            mbatch = 8
            )









# save model
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        

#
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
    #test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(stat['tvl'].dataset), 
        100. * correct / len(stat['tvl'].dataset)))
    

        
#test initial
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

##
# train_starting
def train_epoch(network, stat, optimizer):
    network.train()
    for batch_idx, (data, target) in enumerate( stat['sdl'] ):
        optimizer.zero_grad()
        data, target = data.to(stat['dev']), target.to(stat['dev'])
        output = F.log_softmax(  network(data) )
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #if batch_idx % log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( 
         #       (epoch+1), batch_idx * len(data), len(stat['dl'][1].dataset), 
        #        100. * batch_idx / len(stat['dl'][1]), loss.item()))
    
#
def test(stat, network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in stat['svl']:
            data, target = data.to(stat['dev']), target.to(stat['dev'])
            #output = network(data)
            output = F.log_softmax(  network(data) )
            test_loss += float( F.nll_loss(output, target, size_average=False).item() )
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(stat['svl'].dataset)
    #stat['source_accu'][itr].append( 100. * correct / len(stat['svl'].dataset) )
    #test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(stat['svl'].dataset), 
        100. * correct / len(stat['svl'].dataset)))
 
####
def data_iter( stat, cp):
    ns, nt = len( stat['source'] ), len( stat['target'] )
    sp = ns * cp   
    batch_size = stat['bsize']
    num_examples = ns
    indices = list(range(num_examples)) 
    random.shuffle(indices)     
    # 样本的读取顺序是随机的 
    #j = np.array(indices[i: min(i + batch_size, num_examples)])
    
    for i in range(0, num_examples, batch_size):
        #t += 1
        #stat['la'][t] = t/stat['T'] +0. 
        j = list( indices[i: min(i + batch_size, num_examples)] )
        #yield features.take(j), labels.take(j) 
        batch = torch.utils.data.Subset(stat['source'], j)
        print(len(batch))
        for k in range( batch_size  ):
            xs, ys = batch[k]    
            xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
            
            #print( sp[ j[k] ] )
            nk = int(np.random.choice(a=nt, size=1, replace=False, p=sp[ j[k] ] ))
            xt, yt = stat['target'][nk]
            xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev']) 
            #x = (1 - stat['la'][t]) * xs + stat['la'][t] * xt
            if k == 0:
                image_s, image_t, label_s, label_t = xs, xt, ys, yt
            else:
                image_s, image_t, label_s, label_t = torch.cat((image_s, xs), 0
                                                              ), torch.cat((image_t, xt), 0
                                                                          ), torch.cat((label_s, ys), 0
                                                                                    ), torch.cat((label_t, yt), 0)
        
        yield image_s, label_s, image_t, label_t
        
####
def transfer(itr, t, network, optimizer, stat, epoch):  
    criterion = nn.CrossEntropyLoss()
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    cp = stat['cp'][itr]
    for d in data_iter(stat, cp):   
        t += 1
        stat['la'][t] = t/ stat['T'] +0. 
        #####evaluating predictions for each pair linear conbinations
        network.eval()       
        with torch.no_grad():
            for m in range( ns ): 
                for n in range( nt ):
                    xs, ys = stat['source'][ m ]
                    xt, yt = stat['target'][ n ]
                    xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
                    xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev'])
                    x = (1 - stat['la'][t]) * xs + stat['la'][t] * xt
                    stat['pred'][ (m,n) ] = F.softmax(  network(x) )
                    
        ####SGD updates
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
        
        ####re-evaluating predictions for each pair linear conbinations
        network.eval()       
        with torch.no_grad():
            for m in range( ns ): 
                for n in range( nt ):
                    xs, ys = stat['source'][ m ]
                    xt, yt = stat['target'][ n ]
                    xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
                    xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev'])
                    x = (1 - stat['la'][t]) * xs + stat['la'][t] * xt
                    stat['re_pred'][ (m,n) ] = F.softmax(  network(x) )
        
                    kl = F.relu(F.kl_div( stat['re_pred'][ (m,n) ].log(), 
                                  stat['pred'][ (m,n) ], None, None, 'sum'))
                    stat['r_dist'][itr][m][n] += float( math.sqrt( kl ) / stat['T'] )
    return t, stat

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
        t, stat =  transfer2(itr, t, network, optimizer, stat, epoch)              
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
    
    cost = stat['r_dist'][itr] + 0. 
    cp = stat['cp'][itr] + 0.     
    #reg1 = 1e-3
    #reg2 = 1e-1 * ( (1.25) ** itr )
    #reg2 =( 5e-2 )* ( 2 ** itr )
    
    #reg =  2 ** itr 
    reg =  2 ** ( itr - 1 )      
    def f(G):
        return - ( G *  cp).sum()
    
    def df(G):
        return - cp 
    
    #stat['cp'][( itr + 1 )] = ot.optim.gcg(ps, pt, cost, reg1, reg2, f, df, verbose=True)
    stat['cp'][( itr + 1 )] = ot.optim.cg(ps, pt, cost, reg, f, df, G0 = cp, verbose=True)
    #cost = stat['tr_loss'][itr] + 0. 
    #cp = ot.emd(ps, pt, cost)
    #stat['cp'][( itr + 1 )] = 0.1 *  ( ot.emd(ps, pt, cost) + 0. ) + 0.9 * ( stat['cp'][itr] + 0.) 
    #stat['cp'][( itr + 1 )] = ot.emd(ps, pt, cost) + 0.
    #copy down the data
    saving['cp'][itr]= stat['cp'][itr]
    saving['loss'][itr]  = stat['loss'][itr] 
    saving['r_dist'] = stat['r_dist']
    print(t, 'T')
    print('Time used is ', time.time() - start)
    print('itr_end' )    
    
######
def transfer3(itr, t, network, optimizer, stat, epoch):  
    criterion = nn.CrossEntropyLoss()
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    cp = stat['cp'][itr]
    for d in data_iter(stat, cp):   
        t += 1
        #stat['la'][t] = t/ stat['T'] +0. 
        ll = t/ stat['T']
        ll_1 = 1 - t/ stat['T']
        stat['la'][t] = np.random.beta( a = t/ stat['T'], b = (1 - t/ stat['T'] + 1e-8) )
        if ll <= 0 or ll_1 <=0:
            print(ll, ll_1, 'ab')
        ####SGD updates
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
        
        ####evaluating predictions for each pair linear conbinations
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
                         #                                                          ) + stat['la'][t] * criterion(outputs, yt))
                        
                        ##### p_{ w_t } ( y | x_{ t } ) 
                        kt = int( t/ interval )
                        stat['pred'][ ( kt, m, n) ] = F.softmax(  network(x) )
                        
                        #### p_{ w_t } ( y | x_{ t - interval } ) ##
                        x1 = (1 - stat['la'][t - interval]) * xs + stat['la'][t - interval ] * xt
                        pred1 = F.softmax(  network(x1) )
                        
                        #### call p_{ w_{ t - interval } } ( y | x_{ t - interval } )
                        pred2 = stat['pred'][ ( kt - 1, m, n) ] 
                        
                        ####KL divergence
                        kl = F.relu(F.kl_div( pred1.log(), 
                                      pred2, None, None, 'sum'))
                        stat['r_dist'][itr][m][n] += float( ( math.sqrt( kl ) * interval ) / stat['T'] )
    return t, stat

def transfer_mixture(itr, t, network, optimizer, stat, epoch):  
    criterion = nn.CrossEntropyLoss()
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    for (ds, dt) in zip ( stat['sdl'], stat['tdl'] ):  
        t += 1
        stat['la'][t] = t/ stat['T'] +0. 
                          
        ####SGD updates
        network.train()
        xs, ys = ds
        xt, yt = dt
        xs, ys = xs.to(stat['dev']), ys.to(stat['dev']) 
        xt, yt = xt.to(stat['dev']), yt.to(stat['dev'])
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = (1 - stat['la'][t]) * criterion(network( xs ), ys) + stat['la'][t] * criterion(network( xt ), yt)
        loss.backward()
        optimizer.step()
        
        stat['loss'][itr].append(float(loss ))
        
        ####evaluating predictions for each pair linear conbinations
        interval = stat['interval']
        if t % interval == 0:
            network.eval()       
            with torch.no_grad():
                kl = 0
                for m in range( ns ): 
                    xs, ys = stat['source'][ m ]
                    xt, yt = stat['target'][ m ]
                    xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
                    xt, yt = xt.unsqueeze(0).to(stat['dev']), torch.tensor(yt).view(-1).to(stat['dev'])

                    ###loss###
                    #outputs = network( x )                        
                    #stat['tr_loss'][itr][m][n] += float( (1 - stat['la'][t]) * criterion(outputs, ys
                    #                                                           ) + stat['la'][t] * criterion(outputs, yt))

                    ##### p_{ w_t } ( y | x_{ t } ) 
                    kt = int( t/ interval )
                    stat['pred_s'][ ( kt, m) ] = F.softmax(  network(xs) )
                    stat['pred_t'][ ( kt, m) ] = F.softmax(  network(xt) )

                    #### p_{ w_t } ( y | x_{ t - interval } ) ##
                    #x1 = (1 - stat['la'][t - interval]) * xs + stat['la'][t - interval ] * xt
                    pred1_s, pred1_t = stat['pred_s'][ ( kt, m) ], stat['pred_t'][ ( kt, m) ]

                    #### call p_{ w_{ t - interval } } ( y | x_{ t - interval } )
                    pred2_s, pred2_t = stat['pred_s'][ ( kt - 1, m) ], stat['pred_t'][ ( kt - 1, m) ]

                    ####KL divergence
                    kl += (1 - stat['la'][t - interval]) * math.sqrt(
                        F.relu(
                            F.kl_div( 
                                pred1_s.log(), pred2_s, None, None, 'sum'
                            )
                        ) 
                    ) + stat['la'][t - interval] * math.sqrt( 
                        F.relu(
                            F.kl_div( 
                                pred1_t.log(), pred2_t, None, None, 'sum'
                            )
                        ) 
                    )
                #stat['r_dist'][itr] += float( ( math.sqrt( kl / ns ) * interval ) / stat['T'] )
                stat['r_dist'][itr] += float( ( kl * interval /ns ) / stat['T'] )
    return t, stat

####
def projection_mixture(network, MNIST_tran_ini, stat, saving, itr):
    optimizer = optim.SGD( network.parameters()
                          , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay'] )
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    #cp = stat['cp'][itr]
    stat['source_accu'][itr] = []
    stat['target_accu'][itr] = []
    stat['loss'][itr] = [] 
    stat['r_dist'][itr] = 0
    #stat['tr_loss'][itr] = np.zeros( (ns ,  nt)  )
    print(itr,'itr')
    
    ####transfer block#####    
    start = time.time()
    t = 0 
    for epoch in range(stat['n_epochs']):
        t, stat =  transfer_mixture(itr, t, network, optimizer, stat, epoch)              
        if (epoch+1) % 2 == 0 or epoch == 0:
            print('#####source loss######')
            test_source(stat, network, itr)
            print('#####target loss######')
            test_target(stat, network, itr)
            print('Time used is ', time.time() - start)
    print('################')
    saving['r_dist'] = stat['r_dist']
    print(t, 'T')
    print('Time used is ', time.time() - start)
    print('itr_end' )    
    
###
######
def transfer2(itr, t, network, optimizer, stat, epoch):  
    criterion = nn.CrossEntropyLoss()
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    cp = stat['cp'][itr]
    for d in data_iter(stat, cp):   
        t += 1
        #stat['la'][t] = t/ stat['T'] +0. 
        ll = t/ stat['T']
        ll_1 = 1 - t/ stat['T']
        stat['la'][t] = np.random.beta( a = t/ stat['T'], b = (1 - t/ stat['T'] + 1e-8) )
        if ll <= 0 or ll_1 <=0:
            print(ll, ll_1, 'ab')
        ####SGD updates
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
        
        ####evaluating predictions for each pair linear conbinations
        interval = stat['interval']
        if t % interval == 0:
            network.eval()   
            dsize = stat['dsize']
            with torch.no_grad():
                #for data in testloader:
                k1 = 0 
                for isource, ds in enumerate( stat['svl'] ):
                    k2 = 0
                    xs, ys = ds
                    xs, ys = xs.to(stat['dev']).unsqueeze(1), ys.to(stat['dev']).unsqueeze(1)
                    for itarget, dt in enumerate( stat['tvl'] ):
                        xt, yt = dt
                        xt, yt = xt.to(stat['dev']).unsqueeze(0), yt.to(stat['dev']).unsqueeze(0)
                        xmix = (xs.repeat(1, dsize, 1, 
                                          1, 1)).mul( 1 - stat['la'][t] ) + (xt.repeat(dsize, 1, 1, 1, 1)).mul( stat['la'][t] )
                        #ys, yt = ys.repeat(1, dsize), yt.repeat(dsize, 1)
                        xmix = xmix.view( -1, 3, 32, 32 )

                        ##### computing p_{ w_t } ( y | x_t ) 
                        kt = int( t/ interval )
                        stat['pred'][ ( kt, k1, k2) ] = F.softmax(  network(xmix) ).cpu()

                        #### computing p_{ w_t } ( y | x_{ t - interval } ) ##
                        x1 = (1 - stat['la'][t - interval]) * xs + stat['la'][t - interval ] * xt
                        xmix1 = (xs.repeat(1, dsize, 1, 
                                          1, 1)).mul( 
                            1 - stat['la'][t - interval] ) + (xt.repeat(dsize, 1, 1, 1, 1)).mul( stat['la'][t- interval] )    
                        xmix1 = xmix1.view( -1, 3, 32, 32 )
                        pred1 = F.softmax(  network(xmix1) )

                        #### call p_{ w_{ t - interval } } ( y | x_{ t - interval } )
                        pred2 = stat['pred'][ ( kt - 1, k1, k2) ].to(stat['dev'])

                        ####KL divergence KL( p_{ w_{ t - interval } } ( y | x_{ t - interval } ) || p_{ w_t } ( y | x_{ t - interval } ) )
                        kl = (pred2 * ( pred2.log() - pred1.log() )).sum(1).view( -1, dsize )
                        #kl = torch.sqrt( F.relu(kl) )
                        #kl = kl * interval / stat['T'] 
                        
                        ###catche the kl increments
                        if k2 == 0:
                            row = kl
                        else:
                            row = torch.cat(( row, kl ) , 1)                        
                        k2 += 1  
                    if k1 == 0:
                        inc = row
                    else:
                        inc = torch.cat(( inc, row ) , 0) 
                    k1 += 1
                #stat['r_dist'][itr]+= float( inc ) 
                #stat['r_dist'][itr] = inc.cpu().numpy()   
                stat['r_dist'][itr] += ( ( torch.sqrt( F.relu(inc) ) * interval ) / stat['T']  ).cpu().numpy()             
    return t, stat
                
                    #for i1 in range(dsize):
                        #    m = int( k1 * dsize + i1 )
                        #    for i2 in range(dsize):
                        #        n = int( k2 * dsize + i2 )
                        #        count = int( i1 * dsize + i2 )
                        #        stat['r_dist'][itr][m][n] += float( ( math.sqrt( F.relu(kl[count]) ) * interval ) / stat['T'] ) 
 