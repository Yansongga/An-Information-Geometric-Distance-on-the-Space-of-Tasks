import torch 
import torchvision as thv
from torchvision import transforms
import  torch as th
from torch.utils.data import DataLoader
#from model import CNet
from utils import  check_mkdir
from utils import  train_epoch, data_iter, transfer, projection
from utils import  test_target, test_source, test
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
#from Res_model import ResNet50, Identity, fcNet, CNN, CNet_torch
from model import CNN_torch, CNN, Net
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


learning_rate = 1e-3
train_size = 200
stat = defaultdict(dict)
stat[ 'n_epochs' ] = 40
stat['bsize'] = 2
stat['dsize'] = 50
stat['weight_decay'] = 5e-4
stat['iterations'] = 8 #num for itrs for couplings
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
        index = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
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
    train0, _ = torch.utils.data.random_split(train, [train_size, len(train)- train_size ])  
    if i == 0:
        stat['source'] = train0
    else:
        stat['target'] = train0

# In[6]:
# define train loader and validation loader
stat['svl'] = DataLoader( stat['source'], batch_size=stat['dsize'], 
                             shuffle=False, drop_last = False)
stat['tvl'] = DataLoader( stat['target'], batch_size=stat['dsize'], 
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
    
images, labels = stat['target'][22]
# show images
imshow(thv.utils.make_grid(images))

# In[8]:
#####probe network for computing image embeeddings \phi
import torchvision.models as models_t
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
print( stat['embedding']  )    
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

# In[9]:
# pre train model on source task and save the model
network = CNN().to(stat['dev'])
optimizer = optim.SGD( network.parameters()
                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay']
)
for epoch in range( 30 ):
    train_epoch(network, stat, optimizer)
    if (epoch + 1) %5 == 0:
        test(stat, network )
####saving pretrained model
torch.save(
    network.state_dict(), 
                   os.path.join(MNIST_tran_ini, 
                               'CNN={}.pth'.format( 'animal' )
                               )
)  

# In[10]:
####loading pre-trained model
network = CNN().to(stat['dev'])
network.load_state_dict(
    torch.load(
        os.path.join(
            MNIST_tran_ini, 'CNN={}.pth'.format( 'animal' )
        )))

optimizer = optim.SGD( network.parameters()
                      , lr=1e-3, momentum=0.9, weight_decay = stat['weight_decay']
)

# In[11]:
###set up before transfer learning
stat['T'] = int(( len(stat['source']) / stat['bsize']) * stat[ 'n_epochs' ]) 
stat['interval'] = int( stat['T'] / 50) 
stat['la'][0] = 0

######### p_{w_0}( y| x ) on source task
start = time.time()
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
                              1, 1)).mul( 1 - stat['la'][0] ) + (xt.repeat(dsize, 1, 1, 1, 1)).mul( stat['la'][0] )     
            xmix = xmix.view( -1, 3, 32, 32 )
            
            ##### computing p_{ w_0 } ( y | x_t ) 
            stat['pred'][ ( 0, k1, k2) ] = F.softmax(  network(xmix) ).cpu()           
            k2 += 1    
        k1 += 1           
print('Time used is ', time.time() - start)


# In[12]:
#couplings updates block
saving = defaultdict(dict)
for itr in range( stat['iterations'] ):
    network = CNN()
    network.load_state_dict(
        torch.load(
            os.path.join(
                MNIST_tran_ini, 'CNN={}.pth'.format( 'animal' )
            )))
    network = network.to(stat['dev'])
    projection(network, MNIST_tran_ini, stat, saving, itr)
    stat[ 'distance' ][ itr ] = torch.tensor( stat['cp'][itr] * stat['r_dist'][itr] ).sum()
    saving['distance'][itr] = stat[ 'distance' ][ itr ]
    print( stat[ 'distance' ], 'riemann distance at ', itr )
    print( torch.tensor( stat['cp'][itr + 1] * stat['r_dist'][itr] ).sum() )
    print( torch.tensor( stat['cp'][itr + 1] * stat['tr_loss'][itr] ).sum(), 'loss' )
    
# In[13]:
######saving experiments
save = './checkpoint'
check_mkdir(save)
states = {
    'statistics': saving                   # 将epoch一并保存
}
#torch.save(states, './checkpoint/1st_CIFAR_ymix{}.t7'.format( (7,4) ))
torch.save( states, './checkpoint/CIFAR-task={}.t7'.format( 
        ( 'animal', 'vehicle', 'ot' )))


