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
from model import ResNet18, ResNet34, ResNet50, Net, CNN
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

 
# train model
def train_epoch(network, stat, optimizer, dl):
    network.train()
    for batch_idx, (data, target) in enumerate( dl ):
        optimizer.zero_grad()
        data, target = data.to(stat['dev']), target.to(stat['dev'])
        output = F.log_softmax(  network(data) )
        loss = F.nll_loss(output, target)
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
 
#### ### self-defined data loader based on the couplings between source and target
def data_iter( stat, cp):
    ns, nt = len( stat['source'] ), len( stat['target'] )
    sp = ns * cp   
    #batch_size = stat['bsize']
    batch_size = int( ns / stat['dl_ratio'] )
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
        #print(len(batch))
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
def projection(network, optimizer, MNIST_tran_ini, stat, saving, mb, itr):
    #optimizer = optim.SGD( network.parameters()
     #                     , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay'] )
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
    print('#####source loss######')
    test_source(stat, network, itr)
    for epoch in range(stat['n_epochs']):
        t, stat =  transfer(mb, itr, t, network, optimizer, stat, epoch)  
        pe =int( stat['p_epoch'] )
        if (epoch+1) % pe == 0 or epoch == 0:
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
def transfer(mb, itr, t, network, optimizer, stat, epoch):  
    criterion = nn.CrossEntropyLoss()
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    cp = stat['cp'][itr]
    for d in data_iter(stat, cp):   
        t += 1
        #stat['la'][t] = t/ stat['T'] +0. 
        ll = t/ stat['T']
        ll_1 = 1 - t/ stat['T']
        stat['la'][t] = np.random.beta( a = 0.5 * t/ stat['T'], b = 0.5 * (1 - t/ stat['T'] + 1e-8) )
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
                   
        ####adjusting gradient with trajectary proxy regulizer term
        if mb > 0: 
            proxy = stat['proxy'] 
            num_p = 0
            for p in network.parameters():
                if p.grad is None:
                    continue
                q = ( stat['proxy_trajectory'][( num_p, t )] + 0. ).to( stat['dev'] )
                p.grad = p.grad + proxy * ( p - q ) + 0.
                num_p += 1
        ####adjusting gradient with optimal transportation proxy regulizer term
        if itr > 0: 
            ot_proxy = stat['ot_proxy'] 
            num_p = 0
            for p in network.parameters():
                if p.grad is None:
                    continue
                q = ( stat['p'][( num_p, t )] + 0. ).to( stat['dev'] )
                p.grad = p.grad + ot_proxy * ( p - q ) + 0.
                num_p += 1
        
        #recording NN weight trajectory
        num_p = 0
        for p in network.parameters():
            if p.grad is None:
                continue
            stat['p'][(num_p, t)] = ( p + 0.).cpu()
            num_p += 1
        
        optimizer.step()        
        stat['loss'][itr].append(float(loss ))
        
        ####evaluating predictions for each pair linear conbinations
        interval = stat['interval']
        if t % interval == 0:
            network.eval()   
            #dsize = stat['dsize']
            with torch.no_grad():
                #for data in testloader:
                k1 = 0 
                for isource, ds in enumerate( stat['svl'] ):
                    k2 = 0
                    xs, ys = ds
                    bs1 = len(ys)
                    xs, ys = xs.to(stat['dev']).unsqueeze(1), ys.to(stat['dev']).unsqueeze(1)
                    for itarget, dt in enumerate( stat['tvl'] ):
                        xt, yt = dt
                        bs2 = len(yt)
                        xt, yt = xt.to(stat['dev']).unsqueeze(0), yt.to(stat['dev']).unsqueeze(0)
                        xmix = (xs.repeat(1, bs2, 1, 
                                          1, 1)).mul( 1 - stat['la'][t] ) + (xt.repeat(bs1, 1, 1, 1, 1)).mul( stat['la'][t] )
                        #ys, yt = ys.repeat(1, len(dt)), yt.repeat(len(ds), 1)
                        xmix = xmix.view( -1, 3, 32, 32 )

                        ##### computing p_{ w_t } ( y | x_t ) 
                        kt = int( t/ interval )
                        stat['pred'][ ( kt, k1, k2) ] = F.softmax(  network(xmix) ).cpu()

                        #### computing p_{ w_t } ( y | x_{ t - interval } ) ##
                        x1 = (1 - stat['la'][t - interval]) * xs + stat['la'][t - interval ] * xt
                        xmix1 = (xs.repeat(1, bs2, 1, 
                                          1, 1)).mul( 
                            1 - stat['la'][t - interval] ) + (xt.repeat(bs1, 1, 1, 1, 1)).mul( stat['la'][t- interval] )    
                        xmix1 = xmix1.view( -1, 3, 32, 32 )
                        pred1 = F.softmax(  network(xmix1) )

                        #### call p_{ w_{ t - interval } } ( y | x_{ t - interval } )
                        pred2 = stat['pred'][ ( kt - 1, k1, k2) ].to(stat['dev'])

                        ####KL divergence KL( p_{ w_{ t - interval } } ( y | x_{ t - interval } ) || p_{ w_t } ( y | x_{ t - interval } ) )
                        kl = (pred2 * ( pred2.log() - pred1.log() )).sum(1).view( -1, bs2 )
                        #print(kl.shape)
                        #print(len(yt), 'yt')
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
                #print(inc.shape, 'inc')
                stat['r_dist'][itr] += ( ( torch.sqrt( F.relu(inc) ) * interval ) / stat['T']  ).cpu().numpy()             
    return t, stat
                
######
# In[8]:
#####probe network for computing image embeeddings \phi
def embedding(stat):
    res = models_t.resnet50(pretrained=True)
    probe = Net( res ).to(stat['dev'])
    ####cosine similarities
    start = time.time()
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    stat['embedding'] = np.zeros( (ns ,  nt)  )
    probe.eval()
    nk = 8 
    with torch.no_grad():
        for kk in range(nk): 
            ####secondly comoute the target images embedding
            k2 = 0
            for itarget, dt in enumerate( stat['tvl'] ):        
                xt, yt = dt
                xt = xt.to(stat['dev'])
                embedding = probe( xt )
                if k2 == 0:
                    zt = embedding
                else:
                    zt = torch.cat(( zt, embedding ) , 0)
                k2 += 1
            ####first compute the source images embedding
            k1 = 0
            for isource, ds in enumerate( stat['svl'] ):        
                xs, ys = ds
                xs = xs.to(stat['dev'])
                embed = probe( xs )
                if k1 == 0:
                    zs = embed
                else:
                    zs = torch.cat(( zs, embed ) , 0)
                k1 += 1
            zs, zt = zs.unsqueeze(1).repeat( 1, nt, 1 ), zt.unsqueeze(0).repeat( ns, 1, 1 )
            cos = nn.CosineSimilarity(dim=2, eps=1e-8)
            dist = - cos( zs, zt)
            stat['embedding'] += dist.cpu().numpy() 
        stat['embedding'] /= nk
    #print( stat['embedding']  )    
    print('Time used is ', time.time() - start)

    # In[8]:
    ##using pot pack to compute warm up initial guess Gamma_0
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

    ###display initial guess couplings
    ot.plot.plot1D_mat(ps, pt, stat['cp'][0 ], 'OT matrix Entrop. reg')
    

#######
def pw0(stat, network):
    start = time.time()
    network.eval()   
    #dsize = stat['dsize']
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
                xmix = (xs.repeat(1, len(yt), 1, 
                                  1, 1)).mul( 1 - stat['la'][0] ) + (xt.repeat(len(ys), 1, 1, 1, 1)).mul( stat['la'][0] )     
                xmix = xmix.view( -1, 3, 32, 32 )

                ##### computing p_{ w_0 } ( y | x_t ) 
                stat['pred'][ ( 0, k1, k2) ] = F.softmax(  network(xmix) ).cpu()           
                k2 += 1    
            k1 += 1           
    print('Time used is ', time.time() - start)
    

########
def data_split(train, index):
    train.targets = torch.tensor( train.targets )
    for k in range( len(index) ):
        if k == 0:
            idx = train.targets == index[k]
        else:
            idx += train.targets == index[k]
    train.targets= train.targets[idx]
    train.data = train.data[idx.numpy().astype(np.bool)]
    return train

#######
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
######
#mini_batched data
def minibatch_data( stat, source_data, target_data ):
    #train_size = stat['train_size']
    ratio = stat['mb_ratio']
    vr = stat['vl_ratio']
    dr = stat['dl_ratio']
    b1, b2 = int(len( source_data) / ratio ), int(len( target_data) /ratio)
    stat['source'], _ = torch.utils.data.random_split(
        source_data, [b1, len(source_data )- b1 ]) 
    stat['target'], _ = torch.utils.data.random_split(
        target_data, [b2, len(target_data)- b2 ]) 
    
    # In[6]:
    # define train loader and validation loader
    stat['svl'] = DataLoader( stat['source'], batch_size= int( b1 / vr ), 
                                 shuffle=False, drop_last = False)
    stat['tvl'] = DataLoader( stat['target'], batch_size= int( b2 / vr ), 
                                 shuffle=False, drop_last = False)

    #stat['sdl'] = DataLoader( stat['source'], batch_size=stat['bsize'], 
    #                             shuffle=True, drop_last = False)
    #stat['tdl'] = DataLoader( stat['target'], batch_size=stat['bsize'], 
    #                             shuffle=True, drop_last = False)
    stat['sdl'] = DataLoader( stat['source'], batch_size=int( b1 /dr ), 
                                 shuffle=True, drop_last = False)
    stat['tdl'] = DataLoader( stat['target'], batch_size=int( b2 /dr), 
                                 shuffle=True, drop_last = False)

    # In[7]:
    ##display images###  
    images, labels = stat['target'][22]
    # show images
    imshow(thv.utils.make_grid(images))

########
def CNN_loading( stat, MNIST_tran_ini):
    network = CNN().to(stat['dev'])
    network.load_state_dict(
        torch.load(
            os.path.join(
                MNIST_tran_ini, 'CNN={}.pth'.format( stat['task1'] )
            )))

    optimizer = optim.SGD( network.parameters()
                          , lr=1e-3, momentum=stat['momentum'], weight_decay = stat['weight_decay']
    )
    return network, optimizer

#######
####
def projection_mix(network, optimizer, MNIST_tran_ini, stat, saving, itr):
    #optimizer = optim.SGD( network.parameters()
    #                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay'] )
    ns, nt = len( stat['source'] ), len( stat['target'] )   
    #cp = stat['cp'][itr]
    stat['source_accu'][itr] = []
    stat['target_accu'][itr] = []
    stat['loss'][itr] = [] 
    stat['r_dist'][itr] = 0
    print(itr,'itr')
    
    ####transfer block#####    
    start = time.time()
    t = 0 
    print('#####source loss######')
    test_source(stat, network, itr)
    for epoch in range(stat['n_epochs']):
        t, stat =  transfer_mix(itr, t, network, optimizer, stat, epoch)              
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
                    
                stat['r_dist'][itr] += float( ( kl * interval  ) / stat['T'] )                             
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
def minibatch_data_mix( stat, source_data, target_data ):
    #train_size = stat['train_size']
    #stat['source'], _ = torch.utils.data.random_split(
    #    source_data, [train_size, len(source_data )- train_size ]) 
    #stat['target'], _ = torch.utils.data.random_split(
    #    target_data, [train_size, len(target_data)- train_size ]) 
    stat['source'], stat['target'] = source_data, target_data

    # In[6]:
    # define train loader and validation loader
    stat['svl'] = DataLoader( stat['source'], batch_size=int( len( stat['source'] )/10 ), 
                                 shuffle=False, drop_last = False)
    stat['tvl'] = DataLoader( stat['target'], batch_size=int( len( stat['target'] )/10 ), 
                                 shuffle=False, drop_last = False)

    stat['sdl'] = DataLoader( stat['source'], batch_size=int( len( stat['source'] )/1000 ), 
                                 shuffle=True, drop_last = False)
    stat['tdl'] = DataLoader( stat['target'], batch_size=int( len( stat['target'] )/1000 ), 
                                 shuffle=True, drop_last = False)

    # In[7]:
    ##display images###  
    images, labels = stat['target'][22]
    # show images
    imshow(thv.utils.make_grid(images))
