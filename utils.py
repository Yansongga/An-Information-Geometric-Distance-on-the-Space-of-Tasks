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
        for data, target in stat['tdata_vl']:
            data, target = data.to(stat['dev']), target.to(stat['dev'])
            #output = network(data)
            output = F.log_softmax(  network(data) )
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(stat['tdata_vl'].dataset)
    #stat['target_accu'][itr].append( 100. * correct / len(stat['tvl'].dataset) )
    #test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(stat['tdata_vl'].dataset), 
        100. * correct / len(stat['tdata_vl'].dataset)))
            
#source dataset validation
def test_source(stat, network, itr):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in stat['sdata_vl']:
            data, target = data.to(stat['dev']), target.to(stat['dev'])
            #output = network(data)
            output = F.log_softmax(  network(data) )
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(stat['sdata_vl'].dataset)
    #stat['source_accu'][itr].append( 100. * correct / len(stat['svl'].dataset) )
    #test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(stat['sdata_vl'].dataset), 
        100. * correct / len(stat['sdata_vl'].dataset)))

 
# train model
def train_epoch(network, stat, optimizer, dl):
    network.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate( dl ):
        optimizer.zero_grad()
        data, target = data.to(stat['dev']), target.to(stat['dev'])
        #output = F.log_softmax(  network(data) )
        #loss = F.nll_loss(output, target)
        output =  network(data)
        loss = criterion(output, target)
        loss.backward()
        #for p in network.parameters():
        #    if p.grad is None:
        #        continue
        #    p.grad = p.grad * 100 + 0. 
        optimizer.step()
   
    
# test model
def test(stat, network, vl):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in vl:
            data, target = data.to(stat['dev']), target.to(stat['dev'])
            #output = network(data)
            output = F.log_softmax(  network(data) )
            test_loss += float( F.nll_loss(output, target, size_average=False).item() )
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(vl.dataset)
    #stat['source_accu'][itr].append( 100. * correct / len(stat['svl'].dataset) )
    #test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(vl.dataset), 
        100. * correct / len(vl.dataset)))
    return 100. * correct / len(vl.dataset)
 
#### ### self-defined data loader based on the couplings between source and target
def data_iter( stat, itr):
    block_size = stat['block_size']
    bsize = stat['dl_size']
    num_examples = stat['batch_size']
    indices = list(range(num_examples)) 
    random.shuffle(indices)     
    # 样本的读取顺序是随机的 
    #j = np.array(indices[i: min(i + batch_size, num_examples)])    
    for i in range(0, num_examples, bsize):
        #t += 1
        #stat['la'][t] = t/stat['T'] +0. 
        j = list( indices[i: min(i + bsize, num_examples)] )
        #yield features.take(j), labels.take(j) 
        batch = torch.utils.data.Subset(stat['s_data'], j)
        #print(len(batch))
        for k in range( bsize  ):
            xs, ys = batch[k]    
            xs, ys = xs.unsqueeze(0).to(stat['dev']), torch.tensor(ys).view(-1).to(stat['dev']) 
            ####check block id and sub id
            block_id = int( j[k] / block_size )
            sub_id = j[k] - block_id * block_size
            
            #print( sp[ j[k] ] )
            sp = block_size * stat['cp'][itr][block_id]
            nk = int(np.random.choice(a=block_size, size=1, replace=False, p=sp[ sub_id ] ))
            xt, yt = stat['target'][ block_id ][nk]
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
def projection(network, optimizer, tl, MNIST_tran_ini, stat, saving, mb, itr):
    block_size = stat['block_size']
    for block_id in range( stat['num_block'] ):
        stat['r_dist'][(itr, block_id)] = np.zeros( (block_size, block_size)  )
    print(itr,'itr')
    
    ####transfer block#####    
    start = time.time()
    t = 0     
    print('#####source loss######')
    test_source(stat, network, itr)
    for epoch in range(stat['n_epochs']):
        optimizer = optim.SGD( network.parameters()
                              , lr=stat['lr'], momentum=stat['momentum'], weight_decay = stat['weight_decay']
        )
        #if stat['net'] is not 'CNN':
        #    optimizer = optim.SGD( network.parameters()
         #                     , lr= learning_rate( 0.1, epoch), momentum=0.9, weight_decay = 5e-4, nesterov=False
         #   )
         #   print('Wide')
        t, stat =  transfer(mb, itr, t, network, optimizer, stat, epoch)  
        pe =int( stat['p_epoch'] )
        if (epoch+1) % 10 == 0:
            print('Time used is ', time.time() - start, epoch +1, 'epoch')
        if (epoch+1) % pe == 0 or epoch == 0:
            print('###proceding##', epoch +1, '##of##', stat['n_epochs'])
            #print('#####source loss######')
            #test_source(stat, network, itr)
            print('#####target loss######')
            test_target(stat, network, itr)
            print('#####test target loss######')
            test(stat, network, tl )
            print('Time used is ', time.time() - start)
    print('################')
    
    nm = 0.
    for block_id in range( stat['num_block'] ):        
        ps = np.ones( (block_size,) ) / block_size
        pt = np.ones( (block_size, ) )/block_size      
        cost = stat['r_dist'][(itr, block_id)] + 0. 
        cp = stat['cp'][itr][block_id] + 0.     
        
        #reg = 0.1 * (2 **(itr-1))
        #reg = 0.05 * (10 **itr) 
        reg = 0.05 * (5 **itr) 
        def f(G):
            #return 0.5 * np.sum(G**2) - np.sum( G *  cp) 
            return 0.5 * np.sum((G - cp)**2) 

        def df(G):
            return G - cp 

        coupling = ot.optim.cg(ps, pt, cost, reg, f, df, G0 = None, verbose=False)
        stat['cp'][ itr + 1 ].append( coupling )
        nm += f( stat['cp'][ (itr + 1) ][block_id]  )
    stat['norm_list'].append( nm) 
        
        #copy down the data
    saving['cp'][itr]= stat['cp'][itr]
    saving['r_dist'] = stat['r_dist']
    print(t, 'T')
    print('Time used is ', time.time() - start)
    print('itr_end' )    

######
def transfer(mb, itr, t, network, optimizer, stat, epoch):  
    criterion = nn.CrossEntropyLoss()
    for d in data_iter(stat, itr):   
        t += 1
        #interval_load = int(stat['interval']/stat['sub_'])
        interval_load = stat['interval_load']
        t_reminder = int( t % interval_load)
        if t_reminder == 0:
            t_reminder = interval_load
        #stat['la'][t] = t/ stat['T'] +0. 
        ll = t/ stat['T']
        ll_1 = 1 - t/ stat['T']
        stat['la'][t] = np.random.beta( a =  0.5 * t/ stat['T'], b = 0.5 *(1 - t/ stat['T'] + 1e-8) )
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
        
        ####loading the NN weight trajectory piece for last minibatch
        if mb > 1000:
            if t % interval_load == 1:
                loading_mb = torch.load(
                        './checkpoint/mb_trajectory={}.t7'.format( 
                           ( stat['task1'], stat['task2'], mb-1, t, t + interval_load -1 )
                        )
                    )
                stat['proxy_trajectory'] = loading_mb['trajectory']
                
        ####loading the NN weight trajectory piece for last iteration
        if itr > 1000:
            if t % interval_load == 1:
                loading_itr = torch.load(
                        './checkpoint/trajectory={}.t7'.format( 
                           #( stat['task1'], stat['task2'], t, t + interval_load -1 )
                            ( stat['task1'], t, t + interval_load -1 )
                        )
                    )
                stat['p'] = loading_itr['trajectory']
                   
        ####adjusting gradient with trajectary proxy regulizer term
        if mb > 1000: 
            proxy = stat['proxy'] 
            num_p = 0
            for p in network.parameters():
                if p.grad is None:
                    continue
                q = ( stat['proxy_trajectory'][( num_p, t_reminder )] + 0. ).to( stat['dev'] )
                p.grad = p.grad + proxy * ( p - q ) + 0.
                num_p += 1
        ####adjusting gradient with optimal transportation proxy regulizer term
        if itr > 1000:            
            ot_proxy = stat['ot_proxy']          
            num_p = 0
            for p in network.parameters():
                if p.grad is None:
                    continue
                q = ( stat['p'][( num_p, t_reminder )] + 0. ).to( stat['dev'] )
                p.grad = p.grad + ot_proxy * ( p - q ) + 0.
                norm = torch.norm(p -q)
                stat['norm'][itr] += float(((p -q) ** 2).sum())
                num_p += 1
        
        #recording and copy down NN weight trajectory piece for current iteration    
        if mb > 1000:
            num_p = 0
            for p in network.parameters():
                if p.grad is None:
                    continue
                stat['p'][(num_p, t_reminder)] = ( p + 0.).cpu()
                num_p += 1
            if t_reminder == interval_load:
                tj = { 'trajectory': stat['p']  }
                torch.save( tj, './checkpoint/trajectory={}.t7'.format( 
                    #( stat['task1'], stat['task2'], t - interval_load + 1, t )
                    ( stat['task1'], t - interval_load + 1, t )
                )) 

        ####optimizer step
        optimizer.step()    
                
        ####evaluating predictions for each pair linear conbinations
        interval = stat['interval']
        if t % interval == 0:
            network.eval()   
            with torch.no_grad():
                for block_id in range( stat['num_block'] ):
                #for data in testloader:
                    k1 = 0 
                    for isource, ds in enumerate( stat['svl'][ block_id ] ):
                        k2 = 0
                        xs, ys = ds
                        bs1 = len(ys)
                        xs, ys = xs.to(stat['dev']).unsqueeze(1), ys.to(stat['dev']).unsqueeze(1)
                        for itarget, dt in enumerate( stat['tvl'][ block_id ] ):
                            xt, yt = dt
                            bs2 = len(yt)
                            xt, yt = xt.to(stat['dev']).unsqueeze(0), yt.to(stat['dev']).unsqueeze(0)
                            xmix = (xs.repeat(1, bs2, 1, 
                                              1, 1)).mul( 1 - stat['la'][t] ) + (xt.repeat(bs1, 1, 1, 1, 1)).mul( stat['la'][t] )
                            #ys, yt = ys.repeat(1, len(dt)), yt.repeat(len(ds), 1)
                            xmix = xmix.view( -1, 3, 32, 32 )

                            ##### computing p_{ w_t } ( y | x_t ) 
                            kt = int( t/ interval )
                            stat['pred'][ ( kt, k1, k2, block_id) ] = F.softmax(  network(xmix) ).cpu()

                            #### computing p_{ w_t } ( y | x_{ t - interval } ) ##
                            x1 = (1 - stat['la'][t - interval]) * xs + stat['la'][t - interval ] * xt
                            xmix1 = (xs.repeat(1, bs2, 1, 
                                              1, 1)).mul( 
                                1 - stat['la'][t - interval] ) + (xt.repeat(bs1, 1, 1, 1, 1)).mul( stat['la'][t- interval] )    
                            xmix1 = xmix1.view( -1, 3, 32, 32 )
                            pred1 = F.softmax(  network(xmix1) )

                            #### call p_{ w_{ t - interval } } ( y | x_{ t - interval } )
                            pred2 = stat['pred'][ ( kt - 1, k1, k2, block_id) ].to(stat['dev'])

                            ####KL divergence KL( p_{ w_{ t - interval } } ( y | x_{ t - interval } ) || p_{ w_t } ( y | x_{ t - interval } ) )
                            kl = (pred2 * ( pred2.log() - pred1.log() )).sum(1).view( -1, bs2 )

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
                    #print(inc.shape, 'inc')
                    stat['r_dist'][(itr, block_id)] += ( ( torch.sqrt( F.relu(inc) ) * interval ) / stat['T']  ).cpu().numpy()          
    return t, stat
                
######
# In[8]:
#####probe network for computing image embeeddings \phi
def embedding_block(stat):
    stat['cp'][0 ] = [] 
    block_size = stat['block_size']
    bsize = stat['dl_size'] 
    res = models_t.resnet50(pretrained=True)
    probe = Net( res ).to(stat['dev'])
    ####cosine similarities
    start = time.time()
    #ns, nt = len( stat['source'] ), len( stat['target'] )   
    cos = nn.CosineSimilarity(dim=2, eps=1e-8)
    for block_id in range( stat['num_block'] ):
        stat['embedding'][ block_id ] = np.zeros( (block_size ,  block_size)  )
        probe.eval()
        with torch.no_grad():
            #for kk in range(nk): 
            ####secondly comoute the target images embedding
            k2 = 0
            for itarget, dt in enumerate( stat['tvl'][ block_id ] ):        
                xt, yt = dt
                xt = xt.to(stat['dev'])
                embedding = probe( xt )
                if k2 == 0:
                    zt = embedding
                else:
                    zt = torch.cat(( zt, embedding ) , 0)
                k2 += 1
            zt = zt.unsqueeze(0).repeat( bsize, 1, 1 )
            ####first compute the source images embedding
            k1 = 0
            for isource, ds in enumerate( stat['sel'][ block_id ] ):        
                xs, ys = ds
                xs = xs.to(stat['dev'])
                zs = probe( xs )
                zs = zs.unsqueeze(1).repeat( 1, block_size, 1 )
                cos_sim = - cos( zs, zt)                
                if k1 == 0:
                    dist = cos_sim
                else:
                    dist = torch.cat(( dist, cos_sim ) , 0)
                k1 += 1
           
            stat['embedding'][block_id] = dist.cpu().numpy() 
            #stat['embedding'] /= nk
        #print( stat['embedding']  )    
        print('Time used is ', time.time() - start)

        # In[8]:
        ##using pot pack to compute warm up initial guess Gamma_0
        def f(G):
            return np.sum(G * np.log(G))
        def df(G):
            return np.log(G) + 1.

        reg = 1e-3
        ps = np.ones( (block_size,) ) / block_size
        pt = np.ones( (block_size, ) )/block_size
        cost = stat['embedding'][block_id] + 0.
        #stat['cp'][0] = ot.emd(ps, pt, stat['embedding'] ) + 0. 
        coupling = ot.optim.cg(ps, pt, cost, reg, f, df, verbose=False)
        stat['cp'][0 ].append( coupling  )

        ###display initial guess couplings
    ot.plot.plot1D_mat(ps, pt, stat['cp'][0 ][0], 'OT matrix Entrop. reg')
    

#######
def pw0(stat, network):
    start = time.time()
    network.eval()   
    #dsize = stat['dsize']
    with torch.no_grad():
        for block_id in range( stat['num_block'] ):
            #for data in testloader:
            k1 = 0 
            for isource, ds in enumerate( stat['svl'][ block_id ] ):
                k2 = 0
                xs, ys = ds
                xs, ys = xs.to(stat['dev']).unsqueeze(1), ys.to(stat['dev']).unsqueeze(1)
                for itarget, dt in enumerate( stat['tvl'][ block_id ] ):
                    xt, yt = dt
                    xt, yt = xt.to(stat['dev']).unsqueeze(0), yt.to(stat['dev']).unsqueeze(0)
                    xmix = (xs.repeat(1, len(yt), 1, 
                                      1, 1)).mul( 1 - stat['la'][0] ) + (xt.repeat(len(ys), 1, 1, 1, 1)).mul( stat['la'][0] )     
                    xmix = xmix.view( -1, 3, 32, 32 )

                    ##### computing p_{ w_0 } ( y | x_t ) 
                    stat['pred'][ ( 0, k1, k2, block_id) ] = F.softmax(  network(xmix) ).cpu()           
                    k2 += 1    
                k1 += 1           
    print('Time used is ', time.time() - start)
    

########
def data_split(train, index, shift):
    train.targets = torch.tensor( train.targets )
    for k in range( len(index) ):
        if k == 0:
            idx = train.targets == index[k]
        else:
            idx += train.targets == index[k]
    train.targets= train.targets[idx]
    train.data = train.data[idx.numpy().astype(np.bool)]
    
    ####relabel 
    for ik in range( len(index) ):
        for nk in range( len(train) ):
            if train.targets[nk] == torch.tensor(index[ik]):
                train.targets[nk] = torch.tensor( int( ik + shift ) )
            #train[nk] = (x, y)      
    return train

#######
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
######
#mini_batched data
def minibatch_data( stat, source_train, source_val, target_train, target_val ):
    #train_size = stat['train_size']
    b1, b2 = stat['batch_size'], stat['batch_size']
    #stat['bsize_sel'] = 2
    s_list =  random.sample( [ k for k in range(len(source_train))], b1) 
    t_list = random.sample( [ k for k in range(len(target_train))], b2) 
    num_block, block_size = stat['num_block'], stat['block_size']
    stat['source_val'] = torch.utils.data.Subset(source_val, s_list )
    stat['target_val'] = torch.utils.data.Subset(target_val, t_list )
    stat['s_data'] = torch.utils.data.Subset(source_train, s_list )
    stat[ 'sdata_vl' ] = DataLoader( stat['source_val'], batch_size= 400, 
                                 shuffle=False, drop_last = False)
    stat[ 'tdata_vl' ] = DataLoader( stat['target_val'], batch_size= 400, 
                                 shuffle=False, drop_last = False)
    for block_id in range( num_block ):
        start, end = int( block_id * block_size ), int( (block_id + 1) * block_size )
        stat['tl'][ block_id ] = t_list[ start: end ]
        stat['sl'][ block_id ] = s_list[ start: end ]
       
        stat['source'][ block_id ] = torch.utils.data.Subset(source_train, stat['sl'][ block_id ])
        stat['s_val'][ block_id ] = torch.utils.data.Subset(source_val, stat['sl'][ block_id ])
    
        stat['target'][ block_id ] = torch.utils.data.Subset(target_train, stat['tl'][ block_id ])
        stat['t_val'][ block_id ] = torch.utils.data.Subset(target_val, stat['tl'][ block_id ])
    
    # In[6]:
    # define train loader and validation loader           int( b1 / vr )int( b2 / vr )
        stat['svl'][ block_id ] = DataLoader( stat['s_val'][ block_id ], batch_size= stat['svl_size'], 
                                 shuffle=False, drop_last = False)
        stat['tvl'][ block_id ] = DataLoader( stat['t_val'][ block_id ], batch_size= stat['tvl_size'], 
                                 shuffle=False, drop_last = False)
    
        stat['tel'][ block_id ] = DataLoader( stat['t_val'][ block_id ], batch_size= stat['dl_size'], 
                                 shuffle=False, drop_last = False)
        stat['sel'][ block_id ] = DataLoader( stat['s_val'][ block_id ], batch_size= stat['dl_size'], 
                                 shuffle=False, drop_last = False)
    
    x, y = source_val[ s_list[0] ]
    imshow(x)
    x, y = stat['s_val'][0][0]
    imshow(x)
    
    x, y = target_val[ t_list[5] ]
    imshow(x)
    x, y = stat['t_val'][0][5]
    imshow(x)

########
def Wide_loading( stat, MNIST_tran_ini):
    #network = CNN( stat['num_classes'] ).to(stat['dev'])
    network = Wide_ResNet(depth = 16, widen_factor = 4).to(stat['dev'])
    network.load_state_dict(
        torch.load(
            os.path.join(
                #MNIST_tran_ini, 'CNN={}.pth'.format( (stat['task1'], stat['task2']) )
                #MNIST_tran_ini, 'CNN={}.pth'.format( stat['task1'] )            
                MNIST_tran_ini, 'WideRes-16-4={}.pth'.format( stat['task1'] )
            )))
    print('16*4')
    optimizer = optim.SGD( network.parameters()
                          , lr=stat['lr'], momentum=stat['momentum'], weight_decay = stat['weight_decay']
    )
    return network, optimizer

###########
def CNN_loading( stat, MNIST_tran_ini):
    network = CNN( ).to(stat['dev'])
    #network = Wide_ResNet(depth = 16, widen_factor = 4).to(stat['dev'])
    network.load_state_dict(
        torch.load(
            os.path.join(
                #MNIST_tran_ini, 'CNN={}.pth'.format( (stat['task1'], stat['task2']) )
                MNIST_tran_ini, 'CNN={}.pth'.format( stat['task1'] )            
            )))
    print('CNN')
    optimizer = optim.SGD( network.parameters()
                          , lr=stat['lr'], momentum=stat['momentum'], weight_decay = stat['weight_decay']
    )
    return network, optimizer

#######
####
def projection_mix(network, optimizer, tl, MNIST_tran_ini, stat, saving, itr):
    #optimizer = optim.SGD( network.parameters()
    #                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay'] )
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    #cp = stat['cp'][itr]
    stat['source_accu'][itr] = []
    stat['target_accu'][itr] = []
    stat['loss'][itr] = [] 
    stat['r_dist'] = 0.0
    print(itr,'itr')
    
    ####transfer block#####    
    start = time.time()
    t = 0 
    print('#####source loss######')
    test_source(stat, network, itr)
    for epoch in range(stat['n_epochs']):
        t, stat =  transfer_mix(itr, t, network, optimizer, stat, epoch)   
        pe =int( stat['p_epoch'] )
        if (epoch+1) % pe == 0 or epoch == 0:
            print('#####source loss######')
            test_source(stat, network, itr)
            print('#####target loss######')
            test_target(stat, network, itr)
            print('#####test target loss######')
            test(stat, network, tl )
            print('Time used is ', time.time() - start)
    print('################')
    saving['r_dist'] = stat['r_dist']
    print(t, 'T')
    print('Time used is ', time.time() - start)
    print('itr_end' )    
    
###
######

def transfer_mix(itr, t, network, optimizer, stat, epoch):  
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
                m = 0 
                kl = 0 
                for (ds, dt) in zip ( stat['svl'], stat['tvl'] ):  
                    xs, ys = ds
                    xt, yt = dt
                    xs, ys = xs.to(stat['dev']), ys.to(stat['dev']) 
                    xt, yt = xt.to(stat['dev']), yt.to(stat['dev'])
                    
                    ##### p_{ w_t } ( y | x_{ t } ) 
                    kt = int( t/ interval )
                    stat['pred_s'][ ( kt, m) ] = F.softmax(  network(xs) ).cpu()
                    stat['pred_t'][ ( kt, m) ] = F.softmax(  network(xt) ).cpu()
                    
                    #### p_{ w_t } ( y | x_{ t - interval } ) ##    
                    
                    pred1_s, pred1_t = stat['pred_s'][ 
                        ( kt, m) ].to(stat['dev']), stat['pred_t'][ ( kt, m) ].to(stat['dev'])
                    
                    #### call p_{ w_{ t - interval } } ( y | x_{ t - interval } )
                    pred2_s, pred2_t = stat['pred_s'][ 
                        ( kt - 1, m) ].to(stat['dev']), stat['pred_t'][ ( kt - 1, m) ].to(stat['dev'])
                    
                    ####KL divergence                   
                    kl1 = ( torch.sqrt( 
                        F.relu( 
                            ( pred2_s * ( pred2_s.log() - pred1_s.log() ) ).sum(1) 
                        ) ).sum(0) ) / len(stat['source'])  
                    kl2 =( torch.sqrt( 
                        F.relu( 
                            ( pred2_t * ( pred2_t.log() - pred1_t.log() ) ).sum(1) 
                        ) ).sum(0)) / len(stat['target'])
                    kl += (1 - stat['la'][t - interval] ) * kl1 + stat['la'][t - interval] * kl2
                    m += 1
                    
                stat['r_dist'] += float( ( kl * interval  ) / stat['T'] )                             
    return t, stat

def pw0_mix(stat, network):
    start = time.time()
    network.eval()   
    #dsize = stat['dsize']
    with torch.no_grad():        
        #for data in testloader:
        m = 0 
        for (ds, dt) in zip ( stat['svl'], stat['tvl'] ):  
            xs, ys = ds
            xt, yt = dt
            xs, ys = xs.to(stat['dev']), ys.to(stat['dev']) 
            xt, yt = xt.to(stat['dev']), yt.to(stat['dev'])
            ##### p_{ w_t } ( y | x_{ t } ) 
            stat['pred_s'][ ( 0, m) ] = F.softmax(  network(xs) ).cpu()
            stat['pred_t'][ ( 0, m) ] = F.softmax(  network(xt) ).cpu()
            m += 1
              
    print('Time used is ', time.time() - start)
    
######
#mini_batched data
def minibatch_data_mix( stat, source_train, source_val, target_train, target_val ):
    #train_size = stat['train_size']
    #stat['source'], _ = torch.utils.data.random_split(
    #    source_data, [train_size, len(source_data )- train_size ]) 
    #stat['target'], _ = torch.utils.data.random_split(
    #    target_data, [train_size, len(target_data)- train_size ]) 
    stat['source'], stat['target'] = source_train, target_train
    stat['s_val'], stat['t_val'] = source_val, target_val
    
    # In[6]:
    # define train loader and validation loader
    stat['svl'] = DataLoader( stat['s_val'], batch_size=int( 100 ), 
                                 shuffle=False, drop_last = False)
    stat['tvl'] = DataLoader( stat['t_val'], batch_size=int( 100 ), 
                                 shuffle=False, drop_last = False)

    stat['sdl'] = DataLoader( stat['source'], batch_size=int( 25 ), 
                                 shuffle=True, drop_last = False)
    stat['tdl'] = DataLoader( stat['target'], batch_size=int( 25 ), 
                                 shuffle=True, drop_last = False)

    # In[7]:
    ##display images###  
    images, labels = stat['target'][22]
    # show images
    imshow(thv.utils.make_grid(images))
    
#####copy trajectory files
from shutil import copyfile
from sys import exit
def copy_traject( stat, mb ):
    file_num = int( stat['T'] / stat['interval_load'] )
    for period in range( file_num  ):
        #interval_load = int(stat['interval']/5)
        interval_load = stat['interval_load']
        t = int( period * interval_load +1  ) 
        source = './checkpoint/trajectory={}.t7'.format( 
                ( stat['task1'], stat['task2'], t , t + interval_load - 1)
        )
        target = './checkpoint/mb_trajectory={}.t7'.format( 
                ( stat['task1'], stat['task2'], mb, t , t + interval_load - 1)
        )
        # adding exception handling
        try:
            copyfile(source, target)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)
            
 ########
def data_split_no_shift(train, index):
    train.targets = torch.tensor( train.targets )
    for k in range( len(index) ):
        if k == 0:
            idx = train.targets == index[k]
        else:
            idx += train.targets == index[k]
    train.targets= train.targets[idx]
    train.data = train.data[idx.numpy().astype(np.bool)]
    
    ####relabel 
    for ik in range( len(index) ):
        for nk in range( len(train) ):
            if train.targets[nk] == torch.tensor(index[ik]):
                train.targets[nk] = torch.tensor( int( ik + shift ) )
            #train[nk] = (x, y)      
    return train

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum( p.numel() for p in net.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)

def learning_rate( init, epoch):
    optim_factor = 0
    #if(epoch > 160):
    if(epoch > 320):
        optim_factor = 3
    elif(epoch > 240):
        optim_factor = 2
    elif(epoch > 120):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)
