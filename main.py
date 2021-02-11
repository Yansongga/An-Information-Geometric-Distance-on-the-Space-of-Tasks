#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
#import torchvision as thv
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import *
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import random
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import matplotlib.pyplot as plt  
 
#transformation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

#args
args = defaultdict(dict)
args = {
    'herbivores': [15, 19, 21, 31, 38],  
    'carnivores': [3, 42, 43, 88, 97], 
    'vehicles-1':  [8, 13, 48, 58, 90], 
    'vehicles-2': [ 41, 69, 81, 85, 89], 
    'flowers': [ 54, 70, 62, 82, 92 ], 
    'scenes': [ 23, 33, 49, 60, 71 ], 
    'dev': torch.device('cuda: 2' ),
    'iterations': 4, 
    'batch_size': 20,
    'datasize': 2500,  # We use superclassess in CIFAR100, each superclass is consisted of 2500 images 
    'block_size': 25, 
    'epochs': 160,
    'partitions': 100, # partitions * d \tau = 1  
    'lr': 8e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'p_y|x': defaultdict(dict),
    'cp': defaultdict(dict)
}
#saving data path
save_results_path = './results_cifar100'
check_mkdir(save_results_path)

#saving model path
save_models_path = './models_cifar100'
check_mkdir(save_models_path)

#define model
network = Wide_ResNet(depth = 16, widen_factor = 4).to(args['dev'])

#define source and target tasks
source_task, target_task =  'herbivores', 'carnivores' 


# In[4]:


#### loading source domain data
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
eval_dataset = torchvision.datasets.CIFAR100(root='./data', train= True, download=True, transform=transform_test)
#splitting source dataset
test_dataset = torchvision.datasets.CIFAR100(root='./data', train= False, download=True, transform=transform_test)
source_train, source_eval = data_split( train_dataset, args[ source_task ], 0 ), data_split( eval_dataset, args[ source_task ], 0 )
source_test = data_split( test_dataset, args[ source_task ], 0 )
#source task dataloader
sdl = torch.utils.data.DataLoader(dataset=source_train, batch_size=100, shuffle=True)
sel = torch.utils.data.DataLoader(dataset=source_eval, batch_size=500, shuffle=False)
stl = torch.utils.data.DataLoader(dataset=source_test, batch_size=500, shuffle=False)

#### loading target domain data
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
eval_dataset = torchvision.datasets.CIFAR100(root='./data', train= True, download=True, transform=transform_test)
#splitting target dataset
test_dataset = torchvision.datasets.CIFAR100(root='./data', train= False, download=True, transform=transform_test)
target_train, target_eval = data_split( train_dataset, args[ target_task ], 5 ), data_split( eval_dataset, args[ target_task ], 5 )
target_test = data_split( test_dataset, args[ target_task ], 5 )
#target task dataloader
tdl = torch.utils.data.DataLoader(dataset=target_train, batch_size=100, shuffle=True)
tel = torch.utils.data.DataLoader(dataset=target_eval, batch_size=500, shuffle=False)
ttl = torch.utils.data.DataLoader(dataset=target_test, batch_size=500, shuffle=False)

#Block-diagonalize transport couplings.
block_diag( args, source_eval, target_eval )


# In[3]:


# pretrain model on source domain 
for epoch in range( 600 ):
    optimizer = optim.SGD(network.parameters(), lr= learning_rate(0.1, epoch), momentum=0.9, weight_decay=5e-4)
    train_epoch(args, network, optimizer, sdl)
    if (epoch +1)%40 == 0:
        #tesing on source train dataset
        test(args, network, sel)
        #tesing on source test dataset
        test(args, network, stl)
####save model pretrained on source domain
torch.save(
    network.state_dict(), 
                   os.path.join(save_models_path, 
                               'WideResNet={}.pth'.format( ( source_task, target_task ) )  
                               )
)


# In[ ]:


#initialize \Gamma_0
Initializing_coupling(args)
    
#evaluating p_{w(0)}(y|x)
p_w0(args, network)

#
coupled_distance_list = []
for itr in range( args['iterations'] ):
    
    #load the model pretrained on source doamin
    network = Wide_ResNet(depth = 16, widen_factor = 4).to(args['dev'])
    network.load_state_dict(
        torch.load(
            os.path.join(
                save_models_path, 'WideResNet={}.pth'.format( ( source_task, target_task ) )            
            )))
    
    #transfer to target domain
    coupled_distance = coupled_transfer(args, network, source_train, target_train, sel, stl, tel, ttl, itr)
    coupled_distance_list.append( coupled_distance )
    print(
    '\n Coupled_Task_Distance: {:.4f};    Iteration: {}\n'.format(
        coupled_distance, itr +1
    )
     )
  






