import sys
sys.path.append('/workspace/mytoolbox')

import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from hat_datasets.load_data import load_image
#from nn.dcgan import Enet32
from torch.utils.data import TensorDataset,DataLoader
from itertools import cycle
import random
from random import shuffle
#from torchvision import models
import torchvision.models as torch_models
#from models.resnet_noisy import resnet18
from models.resnet import ResNet18
from src.clf2 import Confs0
import torchvision.transforms as transforms
from src.utils import WarmUpLR
'''
I want to try other Map(10->2) ways that different from clf2.py

In this code, we try a linear or non-linear 10-->2 MLP and see the trade-off 
between acc1 and acc2.

'''

#define MLP function 

WARM = 1

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)  # Applying ReLU activation function
        return x


class MLP3(nn.Module):
    def __init__(self, input_size=10, output_size=2, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=2e-1)
        #x = F.leaky_relu(self.fc2(x), negative_slope=2e-1)
        x = F.leaky_relu(self.fc3(x), negative_slope=2e-1)
        return x


 

class BaseTrainer(mt.BaseTrainer):
    def __init__(self):
        mt.BaseTrainer.__init__(self, self)
        
        if self.dataset=='mnist' or self.dataset=='fmnist':
            self.nc = 1
            self.nz = 10
            self.train_pct = 1.0
        elif self.dataset=='cifar10':
            self.nc = 3
            self.nz = 10
            self.batch_size = 128
            self.steal_pct = 0.2
            self.train_pct = 1.0
            self.imagesize = 32
        elif self.dataset=='cifar100':
            self.nz = 100
            self.batch_size = 64
            self.steal_pct = 0.4
            self.train_pct = 1.0
            self.imagesize = 32
        elif self.dataset =='caltech101':
            self.nc= 3
            self.nz= 101
            self.batch_size = 64
            self.imagesize = 224
            self.train_pct = 0.8
            self.steal_pct = 0.2
            self.lr=5e-3
        
        
        #initialize 1(3)-layer MLP
        self.models['L'] = MLP3(self.nz,2).to(self.device)
        self.optims['L'] = torch.optim.Adam(self.models['L'].parameters(), lr=1e-4, betas=(0.5, 0.99))
        print("decoder archtecture,",self.models['L'])
        print("Fixing all parameters")
        for param in self.models['L'].parameters():
            param.requires_grad = False


        
                
        if self.model_arch =='resnet':
            self.models['C']  = ResNet18(num_classes=self.nz).to(self.device)
            #self.models['C'] = torch.compile(model, mode="default")
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            #self.scheduler = ReduceLROnPlateau(self.optims['C'], mode='min', factor=0.1, patience=5, verbose=True)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[50,75], gamma=0.1)
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[100,150], gamma=0.1)
            #self.epochs = 200
            self.epochs = 100
            self.batch_size = 128
        elif self.model_arch =='pre_resnet':
            self.lr = 0.005
            self.models['C']  = torch_models.resnet18(weights='DEFAULT')
            self.models['C'].fc = nn.Linear(512, self.nz)
            self.models['C'] = self.models['C'].to(self.device)
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[6,12], gamma=0.1)
            self.epochs = 20
            self.batch_size = 256

        elif self.model_arch =='pre_resnet34':
            self.models['C']  = torch_models.resnet34(weights='DEFAULT')
            self.models['C'].fc = nn.Linear(512, self.nz)
            self.models['C'] = self.models['C'].to(self.device)
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[6,12], gamma=0.1)
            self.epochs = 20
            self.batch_size = 64

        elif self.model_arch == 'cnn':  
            self.models['C'] = mt.nn.dcgan.Enet32(self.nc, self.nz).to(self.device)
            if self.dataset=='cifar10':
                self.epochs = 75
                self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optims['C'], step_size=25, gamma=0.1, last_epoch=-1)
            elif self.dataset =='mnist' or self.dataset=='fmnist':
                self.epochs = 60
                self.optims['C'] = torch.optim.Adam(self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))
            
        #loading training data
        self.train_loader, self.val_loader, self.test_loader,self.steal_loader = \
            load_image(self.dataset, 1.0, self.train_pct, self.steal_pct, self.batch_size, self.imagesize, None, False)
        
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            self.val_loader =  self.test_loader
        print(f"training size:{len(self.train_loader.dataset)}")
        #loading watermark training data
        print("Loading watermark training data...")
        if self.watermark_loader == None:  
            if self.dataset=='fmnist':
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 0.5, self.batch_size, 32, [8,9], False)  
            elif self.dataset =='svhn':
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 0.5, self.batch_size, 32, [8,9], False)
            elif self.dataset=='cifar10' and self.model_arch=='cnn':
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 0.5, self.batch_size, 32, [8,9], False)
            elif self.dataset=='cifar10' and self.model_arch=='resnet':
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 0.05,self.steal_pct, self.batch_size, 32, [8,9], True)
            elif self.dataset=='cifar100' and (self.model_arch=='resnet'or self.model_arch=='pre_resnet'):
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 0.5,self.steal_pct,self.batch_size, 32, [0,1,2], True)
            elif self.dataset=='caltech101':
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 0.5,self.steal_pct,self.batch_size, 32, [10], True)
            function_mapping = { 'Rotate90': image_rotate,'Grayscale':image_Grayscale,'ColorJitter': image_ColorJitter}

            print('Constructing watermark loader...')
            self.wt_loader = self.loader_transform(self.train_loader2,function_mapping[self.wt_task])
            print("Number of wt images:", len(self.wt_loader.dataset))
        else:
            self.wt_loader = self.watermark_loader
            
        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(self.optims['C'], iter_per_epoch * WARM)

        self.records['acc'] = 0.0
        self.acc1=[]
        self.acc2 =[]
         

    def train(self, epoch):
        
        self.models['C'].train()
        self.models['L'].train()
        
        if self.delta == 0:
            for i,(x1, y1) in enumerate(self.train_loader):
                x1, y1= x1.to(self.device,non_blocking=True), y1.to(self.device,non_blocking=True)
                if self.model_arch =='resnet' and x1.shape[1]==1:
                    x1 = x1.repeat(1,3,1,1)
                scores1 = self.models['C'](x1)
                loss = F.cross_entropy(scores1, y1)
                self.optims['C'].zero_grad()
                loss.backward()
                self.optims['C'].step()
                if epoch <= WARM:
                    self.warmup_scheduler.step()
                if i % 100 == 0:
                    self.logs['Loss'] = loss.item()
                    self.print_logs(epoch, i)
        else:
            for i,((x1, y1),(x2,y2)) in enumerate(zip(self.train_loader,cycle(self.wt_loader))):
                x1, y1= x1.to(self.device,non_blocking=True), y1.to(self.device,non_blocking=True)
                if self.model_arch =='resnet' and x1.shape[1]==1:
                    x1 = x1.repeat(1,3,1,1)
                scores1 = self.models['C'](x1)
                loss1 = F.cross_entropy(scores1, y1)
  
                x2, y2= x2.to(self.device), y2.to(self.device)
                if x2.shape[1]==1 and self.model_arch =='resnet':
                    x2 = x2.repeat(1,3,1,1)
                N = len(x2)
                scores2 = self.models['C'](x2)
                wt_scores = get_wtscore(scores2,self.models['L'],self.wt_bit)
                loss2 = F.cross_entropy(wt_scores, y2)
                loss = loss1 + self.delta * loss2
                #loss = loss1/(loss1.detach()+1e-6) + self.delta*(loss2/(loss2.detach()+1e-6))

                self.optims['C'].zero_grad()
                self.optims['L'].zero_grad()
                loss.backward()
                self.optims['C'].step()
                self.optims['L'].step()

                if epoch <= WARM:
                    self.warmup_scheduler.step()
                if i % 100 == 0:
                    if self.delta!=0:
                        self.logs['Loss1'] = loss1.item()
                        self.logs['Loss2'] = loss2.item()
                    self.logs['Loss'] = loss.item()
                    self.print_logs(epoch, i)


        self.scheduler.step() 
       
        '''
        if epoch>51 and self.delta<1e-5:
            self.delta=0.01
            print(f"currrent epoch:{epoch}, delta:{self.delta}")
        '''
        return loss.item()


    def loader_transform(self,loader,transform):
        X=[]
        Y =[]
        for x, y in loader:
            X.append(x)
            Y.append(torch.zeros_like(y))
            x_rot90 = transform(x)
            #x_rot90 = x.permute(0, 1, 3,2)
            X.append(x_rot90)
            Y.append(torch.ones_like(y))
        X = torch.cat(X,0)
        Y = torch.cat(Y,0) 
        tensordata_rot = TensorDataset(X, Y)
        loader_rot = DataLoader(tensordata_rot, batch_size=self.batch_size, shuffle=True, num_workers=4,drop_last=False)
        return loader_rot
    

                
    def eval(self, epoch):
        if epoch%1 ==0:
            acc1 = eval_task1(self.models['C'],self.val_loader,self.device,self.model_arch)
            self.acc1.append(acc1)
            is_best = False
            if acc1 >= self.records['acc']:
                is_best = True
                self.records['acc'] = acc1
            #if self.partition is not None:
            if self.delta != 0.0:
                acc2,_,_ = eval_task2(self.models['C'],self.models['L'],self.wt_loader,self.wt_bit,self.device,self.model_arch)
                self.acc2.append(acc2)
                print(f'val acc: {acc1:.4f}, trigger acc: {acc2:.3f}')
            else:
                print(f'val acc: {acc1:.4f}')
        else:
            is_best = False
            
        return is_best

def get_wtscore(scores,wt_net,k):
    #find top-k score indicies
    topk_indices = torch.topk(scores, k=k, dim=1)[1]
    masked_score2 = scores.clone()
    zeros_tensor = torch.zeros_like(scores)
    masked_score2.scatter_(1, topk_indices, zeros_tensor)
    score2_mean = torch.mean(masked_score2,dim=1,keepdim=True)
    score2_std = torch.std(masked_score2,dim=1,keepdim=True)
    normalized_score2= (masked_score2 - score2_mean) / score2_std
    wt_scores = wt_net(normalized_score2) # score
    return wt_scores

def image_rotate(image_batch):
    image_batch_rot90 = image_batch.permute(0, 1, 3,2)
    return image_batch_rot90

def image_Grayscale(image_batch):
    #color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    color_transform =transforms.RandomGrayscale(p=1)
    distorted_image_batch = color_transform(image_batch)
    return distorted_image_batch

def image_ColorJitter(image_batch):
    color_transform = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1)
    distorted_image_batch = color_transform(image_batch)
    return distorted_image_batch

def eval_task1(net,test_loader,device,model_arch):
    net.eval()
    correct = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            if model_arch =='resnet' and x.shape[1]==1:
                x = x.repeat(1,3,1,1)
            x, y = x.to(device), y.to(device)
            scores = net(x)
            pred_y = torch.argmax(scores, dim=1)
            correct += torch.sum(pred_y == y).item()
    N = len(test_loader.dataset)
    acc = correct / N
    #print('acc: {}'.format(acc))
    return round(acc,4)

def eval_task2(net,mlp,wt_loader,wt_bit,device,model_arch):
    mlp.eval()
    net.eval()
    correct = 0.0
    with torch.no_grad():
        SCORES = []
        Y=[]
        for x, y in wt_loader:
            if model_arch =='resnet' and x.shape[1]==1:
                x = x.repeat(1,3,1,1)
            x, y = x.to(device), y.to(device)
            N = len(x)
            Y.append(y)
            scores = net(x)
            scores_wt = get_wtscore(scores,mlp,wt_bit)
            soft_scores = F.softmax(scores_wt,dim=1)
            SCORES.append(soft_scores)
            pred_y = torch.argmax(scores_wt, dim=1)
            correct += torch.sum(pred_y == y).item()
    N = len(wt_loader.dataset)
    acc = correct / N
    #print('acc: {}'.format(acc))
    return round(acc,4),torch.cat(SCORES),torch.cat(Y)



class multitask_Trainer(BaseTrainer, Confs0):
    #model_arch:'resnet','cnn'
    #model_type: base,mt,steal,kd,ft
    #watermark_loader: input watermark loader
    #wt_task: 'Rotate90', 'ColorJitter'
    #wt_bit: watermark logits = self.nz - wt_bit
    #delta: balance wt task and the main task
    #manual_seed
    #device_ids: gpu:0 
    def __init__(self,dataset,model_arch,wt_loader,wt_task,wt_bit,delta,manual_seed,device_ids):
        Confs0.__init__(self)
        self.dataset = dataset
        self.model_arch = model_arch
        self.watermark_loader = wt_loader   
        self.device_ids = device_ids
        self.wt_task = wt_task
        self.wt_bit = wt_bit
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.delta = delta
        self.seed = manual_seed
        mt.utils.set_seed(manual_seed)
        print(f"delta:{self.delta}")
        if delta ==0.0:
            self.flag = f'mean-{self.dataset}-{model_arch}-delta{self.delta}-seed{manual_seed}'
        else:

            self.flag = f'{self.dataset}-{model_arch}-wt_task{wt_task}-wtbit{wt_bit}-delta{self.delta}-seed{manual_seed}'
        #self.flag = f'clf-{self.dataset}-{model_arch}-seed{manual_seed}'
        BaseTrainer.__init__(self)


