import os, pdb, sys, json, subprocess,        time, logging, argparse,        pickle, math, gzip, numpy as np,        glob
       #pandas as pd,        
from functools import partial, reduce
from pprint import pprint
from copy import deepcopy

import  torch as th, torch.nn as nn,         torch.backends.cudnn as cudnn,         torchvision as thv,         torch.nn.functional as F,         torch.optim as optim
from torchvision import transforms
cudnn.benchmark = True
import torch

from collections import defaultdict
from torch._six import container_abcs
import torch
from copy import deepcopy
from itertools import chain
from model import *
from torch.utils.data import DataLoader
import math

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import random
import torchvision.models as models_t
import matplotlib.pyplot as plt  
import torch.optim as optim

from tqdm import trange
import time



#60/ 120/ 160/ 200
def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 480):
        optim_factor = 3
    elif(epoch > 360):
        optim_factor = 2
    elif(epoch > 180):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

#train
def train_epoch(args, network, optimizer, dl):
    network.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate( dl ):
        optimizer.zero_grad()
        data, target = data.to(args['dev']), target.to(args['dev'])
        output =  network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

#test function
def test(args, network, testloader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(args['dev']), target.to(args['dev'])
            output = F.log_softmax(  network(data) )
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(testloader.dataset), 
        100. * correct / len(testloader.dataset)))

# save model
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

#Using mixup to interpolate source and target images
def circle_mixup( args ):
    args['tau'] = defaultdict(dict)
    t_range =int( ( args['datasize'] / args['batch_size'] ) * args['epochs'] )
    num_block = int( args['datasize'] / args['block_size']  )
    for t in range( t_range ):
        t += 1
        tau_list = []
        for block_id in range( num_block ):
            if t < t_range:
                tau = np.random.beta( a =  0.5 * t/ t_range, b = 0.5 *(1 - t/ t_range) )
            else:
                tau = 1
            tau_list.append( tau )
        args['tau'][t] = tau_list

def beta_mixup( args ):
    args['tau'] = [ 0 ]
    count_range =int( ( args['datasize'] / args['batch_size'] ) * args['epochs'] )
    for t in range(count_range):
        x = (t+1) / count_range
        if x < 0.5:
            tau = max( (0.5 - np.arcsin( np.sqrt( 0.5 - x ) ) * 2 / math.pi), 0)
            #tau = max ( 0.5 - np.sqrt( 0.25 - x ** 2 ), 0 )
        else:
            tau = max( (1.5 - np.arcsin( np.sqrt( 1.5 - x ) ) * 2 / math.pi), 0 )
            #tau = max(0.5 + np.sqrt( 0.25 - ( 1- x) ** 2 ), 0)
        args['tau'].append( tau )
        
#
def data_split( dataset, labels, shift):
    dataset.targets = torch.tensor( dataset.targets )
    for k in range( len(labels) ):
        if k == 0:
            idx = dataset.targets == labels[k]
        else:
            idx += dataset.targets == labels[k]
    dataset.targets= dataset.targets[idx]
    dataset.data = dataset.data[idx.numpy().astype(np.bool)]
    
    ####re-labelling images
    for k in range( len(labels) ):
        for i in range( len(dataset) ):
            if dataset.targets[i] == torch.tensor(labels[k]):
                dataset.targets[i] = torch.tensor( int( k + shift ) )
            #train[nk] = (x, y)      
    return dataset

#Block-diagonal transport couplings.
def block_diag( args, source_eval, target_eval ):
    beta_mixup( args )
    args['s_eval_loader'] = defaultdict(dict)
    args['t_eval_loader'] = defaultdict(dict)
    num_block = int( args['datasize'] / args['block_size']  )
    for block_id in range( num_block ):
        start, end = int( block_id * args['block_size'] ), int( (block_id + 1) * args['block_size'] )
        image_list = list( range( start, end ) )
        s_block, t_block = torch.utils.data.Subset(source_eval, image_list ), torch.utils.data.Subset(target_eval, image_list )  
        args['s_eval_loader'][ block_id ] = DataLoader( s_block, batch_size= args['block_size'], shuffle=False, drop_last = False)     
        args['t_eval_loader'][ block_id ] = DataLoader( t_block, batch_size= args['block_size'], shuffle=False, drop_last = False) 
                                                               
#Coupled Transfer Distance    
def coupled_transfer(args, network, source_train, target_train, sel, stl, tel, ttl, itr):
    
    block_size = args['block_size']
    num_block = int( args['datasize'] / args['block_size']  )
    args['dist'] = defaultdict(dict)
    for block_id in range( num_block ):
        args['dist'][block_id] = np.zeros( (block_size, block_size)  )
    
    ####transfer block#####    
    start = time.time()
    count = 0   
                                                               
    #checking the model well-pretrained on source domain                                                      
    print('#source domian #')
    test(args, network, sel)
    test(args, network, stl)
    
    #for epoch in range( args['epochs'] ):
    for epoch in trange(args['epochs'] ): 
        time.sleep(1)
        optimizer = optim.SGD( network.parameters()
                              , lr=args['lr'], momentum=args['momentum'], weight_decay = args['weight_decay']
        )
        
        count, args =  transfer_epoch(args, network, optimizer, source_train, target_train, itr, count) 
      
        #if (epoch+1) % 40 == 0:
            #print('Time used is ', time.time() - start, epoch +1, 'epoch', count,'count')
            #print('###proceding##', epoch +1, '##of##', args['epochs'])            
       #     print('target train domain loss is')
        #    test(args, network, tel)
        #    print('target test domain loss is')
        #    test(args, network, ttl)
    
    args['cp'][ itr + 1 ] = []
    info_distance_list = []
    for block_id in range( num_block ):        
        ps = np.ones( (block_size,) ) / block_size
        pt = np.ones( (block_size, ) )/block_size      
        cost = args['dist'][ block_id] + 0. 
        cp = args['cp'][itr][block_id] + 0.     
        
        reg = 0.05 * (5 **itr) 
        def f(G):
            return 0.5 * np.sum((G - cp)**2) 

        def df(G):
            return G - cp 

        coupling = ot.optim.cg(ps, pt, cost, reg, f, df, G0 = None, verbose=False)
        args['cp'][ itr + 1 ].append( coupling )
        info_distance = torch.tensor( args['cp'][itr][block_id] * args['dist'][ block_id] ).sum()
        info_distance_list.append( info_distance )
    #print('Time used is ', time.time() - start)
    return( np.mean( info_distance_list ) )
    
#transfer epoch
def transfer_epoch(args, network, optimizer, source_train, target_train, itr, count):  
    #criterion = nn.CrossEntropyLoss( reduce=False )
    criterion = nn.CrossEntropyLoss()
    for d in data_iter( args, source_train, target_train, itr, count):   
        count += 1
       
        ####SGD updates
        network.train()
        xmix, ys, yt, tau = d
        optimizer.zero_grad()      
        
        # forward + backward + optimize
        outputs = network( xmix )
        loss =   (1 - tau) * criterion(outputs, ys)  + tau * criterion(outputs, yt)
        loss.backward()
            
        ####optimizer step
        optimizer.step()    
                
        ####evaluation of transportation cost
        count_range =int( ( args['datasize'] / args['batch_size'] ) * args['epochs'] )
        d_count = int( count_range/args['partitions'] )
        if count % d_count == 0:
            network.eval()   
            with torch.no_grad():
                
                #transportaton cost for each diagonal blocks
                for block_id in range( int( args['datasize'] / args['block_size']  ) ):
                    for (ds, dt) in zip ( args['s_eval_loader'][ block_id ], args['t_eval_loader'][ block_id ] ):             
                        xs, ys = ds
                        bs1 = len(ys)
                        xs, ys = xs.to(args['dev']).unsqueeze(1), ys.to(args['dev']).unsqueeze(1)
                        xt, yt = dt
                        bs2 = len(yt)
                        xt, yt = xt.to(args['dev']).unsqueeze(0), yt.to(args['dev']).unsqueeze(0)
                        xmix = (xs.repeat(1, bs2, 1, 1, 1)).mul( 1 - tau ) + (xt.repeat(bs1, 1, 1, 1, 1)).mul( tau )
                        xmix = xmix.view( -1, 3, 32, 32 )

                        ##### computing p_{ w_tau } ( y | x_tau ) 
                        m = int( count/d_count )
                        args['p_y|x'][ ( m, block_id) ] = F.softmax(  network(xmix) ).cpu()

                        #### computing p_{ w_tau } ( y | x_{ tau - dtau } ) ##
                        tau1 = args['tau'][ count - d_count ] 
                        xmix1 = (xs.repeat(1, bs2, 1, 1, 1)).mul( 1 - tau1 ) + (xt.repeat(bs1, 1, 1, 1, 1)).mul( tau1 )    
                        xmix1 = xmix1.view( -1, 3, 32, 32 )
                        pred1 = F.softmax(  network(xmix1) )

                        #### call p_{ w_{ tau - dtau } } ( y | x_{ tau - dtau } )
                        pred2 = args['p_y|x'][ ( m - 1, block_id) ].to(args['dev'])

                        ####KL divergence KL( p_{ w_{ t - interval } } ( y | x_{ t - interval } ) || p_{ w_t } ( y | x_{ t - interval } ) )
                        kl_divergence = (pred2 * ( pred2.log() - pred1.log() )).sum(1).view( -1, bs2 )

                        
                    args['dist'][block_id] +=  ( 
                        torch.sqrt( 
                            F.relu(kl_divergence)
                        )/ args['partitions']  
                    ).cpu().numpy()          
    return count, args

#### ### self-defined data loader based on the couplings between source and target
def data_iter( args, source_train, target_train, itr, count):
    block_size = args['block_size']
    bsize = args['batch_size']
    num_examples = args['datasize']
    indices = list(range(num_examples)) 
    random.shuffle(indices)     
    
    # 样本的读取顺序是随机的 
    for i in range(0, num_examples, bsize):
        j = list( indices[i: min(i + bsize, num_examples)] )
        #yield features.take(j), labels.take(j) 
        batch = torch.utils.data.Subset(source_train, j)
     
        for k in range( bsize  ):
            xs, ys = batch[k]    
            xs, ys = xs.unsqueeze(0).to(args['dev']), torch.tensor(ys).view(-1).to(args['dev']) 
            #### block id and sub id
            block_id = int( j[k] / block_size )
            sub_id = j[k] - block_id * block_size
            
            #sampling based on coupling
            q = block_size * args['cp'][itr][block_id]
            nk = int(np.random.choice(a=block_size, size=1, replace=False, p=q[ sub_id ] ))
            
            #corresponding images in target domain
            image_id = int( block_id * block_size + nk )
            xt, yt = target_train[ image_id ]
            xt, yt = xt.unsqueeze(0).to(args['dev']), torch.tensor(yt).view(-1).to(args['dev']) 
            
            #interpolatation of source and target image
            tau = args['tau'][ count + 1]
            xmix = ( 1 - tau ) * xs + tau * xt
            if k == 0:
                image, label_s, label_t= xmix, ys, yt
            else:
                image, label_s, label_t = torch.cat((image, xmix), 0
                                                                          ), torch.cat((label_s, ys), 0
                                                                                    ), torch.cat((label_t, yt), 0)
        
        yield image, label_s, label_t, tau
        
#initializations
def Initializing_coupling(args):
    num_block = int( args['datasize'] / args['block_size']  )
    start = time.time()
    #args['cp'] = defaultdict(dict)
    args['cp'][0] = [] 
    block_size, bsize = args['block_size'], args['batch_size']
    
    #load probe network
    res = models_t.resnet50(pretrained=True)
    probe = Net( res ).to(args['dev'])
    
    ####cosine similarities
    cos = nn.CosineSimilarity(dim=2, eps=1e-8)
    cos_list = []
    for block_id in range( num_block ):
        probe.eval()
        with torch.no_grad():
            for (ds, dt) in zip ( args['s_eval_loader'][ block_id ], args['t_eval_loader'][ block_id ] ):             
                xs, ys = ds
                bs1 = len(ys)
                xs = xs.to(args['dev'])
                xt, yt = dt
                bs2 = len(yt)
                xt = xt.to(args['dev'])
                
                #latent representaion
                zs, zt = probe( xs ), probe( xt )
                zs, zt = zs.unsqueeze(1).repeat( 1, bs2, 1 ), zt.unsqueeze(0).repeat( bs1, 1, 1 )
                cos_sim = - cos( zs, zt) 
                
        cos_list.append( cos_sim.cpu().numpy()  )
        
        #initializing \Gamma_0
        def f(G):
            return np.sum(G * np.log(G))
        def df(G):
            return np.log(G) + 1.

        reg = 1e-3
        ps = np.ones( (block_size,) ) / block_size
        pt = np.ones( (block_size, ) )/block_size
        cost = cos_list[block_id] + 0.
        coupling = ot.optim.cg(ps, pt, cost, reg, f, df, verbose=False)
        args['cp'][0 ].append( coupling  )

        ###display initial guess couplings
    #ot.plot.plot1D_mat(ps, pt, stat['cp'][0 ][0], 'OT matrix Entrop. reg')
    
#evaluating p_{w(0)}(y|x)
def p_w0(args, network):
    #start = time.time()
    tau = 0.
    network.eval()   
    with torch.no_grad():
        for block_id in range( int( args['datasize'] / args['block_size']  ) ):
            for (ds, dt) in zip ( args['s_eval_loader'][ block_id ], args['t_eval_loader'][ block_id ] ):             
                xs, ys = ds
                bs1 = len(ys)
                xs, ys = xs.to(args['dev']).unsqueeze(1), ys.to(args['dev']).unsqueeze(1)
                xt, yt = dt
                bs2 = len(yt)
                xt, yt = xt.to(args['dev']).unsqueeze(0), yt.to(args['dev']).unsqueeze(0)
                xmix = (xs.repeat(1, bs2, 1, 1, 1)).mul( 1 - tau ) + (xt.repeat(bs1, 1, 1, 1, 1)).mul( tau )
                xmix = xmix.view( -1, 3, 32, 32 )

                ##### computing p_{ w_0 } ( y | x_tau ) 
                args['p_y|x'][ ( 0, block_id) ] = F.softmax(  network(xmix) ).cpu()
        
