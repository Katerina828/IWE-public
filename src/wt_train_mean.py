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
#from models.resnet_noisy import resnet18
from models.resnet import ResNet18,ResNet34
import torchvision.models as torch_models
from src.utils import WarmUpLR
import torchvision.transforms as transforms
import time
"""
This code is non-parametric mapping
1.first set logits max(s)==0.0
2.Then, cut s to [a,b] according to given partition strategy
different partition may result in different bias.

"""
WARM =1


class Confs0(mt.BaseConfs):
    def __init__(self):
        super().__init__()
        
    def get_dataset(self):
        self.dataset = 'mnist'
        self.nc = 3
        self.nz = 10
        self.epochs = 50
        self.batch_size= 100
        self.lr=0.01
        #self.lr=5e-3
        
    def get_flag(self):
        self.flag = 'dcgan-{}-clf'.format(self.dataset)
    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        
def get_suffle_index(data_len, seed=0):
    subset_index = [i for i in range(data_len)]
    random.seed(seed)
    shuffle(subset_index)
    return subset_index


def get_train_part(train_loader,pct,batch_size):
    trainset = train_loader.dataset
    subset_index = get_suffle_index(len(trainset))
    sample_len = int(pct * len(trainset))
    sampled_trainset = torch.utils.data.Subset(
        trainset, indices=subset_index[0:sample_len]
    )
    print(f"Sampled trainset size: {len(sampled_trainset)}")
    train_sample_loader = DataLoader(sampled_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_sample_loader
    
    

class BaseTrainer(mt.BaseTrainer):
    def __init__(self):
        mt.BaseTrainer.__init__(self, self)
        
        if self.dataset=='mnist' or self.dataset=='fmnist':
            self.nc = 1
            self.nz = 10
            self.train_pct = 1.0
            self.steal_pct = 0.2
            self.imagesize = 28

        elif self.dataset=='cifar10':
            self.nc = 3
            self.nz = 10
            self.batch_size = 64
            self.lr = self.lr*int(self.batch_size/64)
            self.steal_pct = 0.2
            self.train_pct = 1.0
            self.imagesize = 32
            self.epochs = 100

        elif self.dataset=='cifar100':
            self.nz = 100
            self.batch_size = 64
            self.steal_pct = 0.2
            self.train_pct = 1.0
            self.imagesize = 32
            self.epochs = 100

        elif self.dataset =='caltech101':
            self.nc= 3
            self.nz= 101
            self.batch_size = 128
            self.imagesize = 224
            self.train_pct = 0.8
            self.steal_pct = 0.2
            self.lradj = 1

        if self.model_arch =='resnet':
            #self.models['C']  = torch_models.resnet18(weights='DEFAULT')
            self.models['C']  = ResNet18(num_classes=self.nz).to(self.device)
            #self.models['C'] = torch.compile(model, mode="default")
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            #self.scheduler = ReduceLROnPlateau(self.optims['C'], mode='min', factor=0.1, patience=5, verbose=True)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[50,75], gamma=0.1)
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[100,150], gamma=0.1)
            self.epochs = 100
            #self.epochs = 75
        if self.model_arch =='resnet34':
            self.lr = 0.005
            self.models['C']  = torch_models.resnet34()
            self.models['C'].fc = nn.Linear(512, self.nz)
            self.models['C'] = self.models['C'].to(self.device)
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[50,75], gamma=0.1)
            self.epochs = 100
            
        elif self.model_arch =='pre_resnet':
            self.lr = 0.005
            self.models['C']  = torch_models.resnet18(weights='DEFAULT')
            self.models['C'].fc = nn.Linear(512, self.nz)
            self.models['C'] = self.models['C'].to(self.device)
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[6,12], gamma=0.1)
            self.epochs = 20

        elif self.model_arch =='pre_resnet34':
            self.lr = 0.005
            self.models['C']  = torch_models.resnet34(weights='DEFAULT')
            self.models['C'].fc = nn.Linear(512, self.nz)
            # Freeze all layers except the last linear layer
            '''
            print("Fixing former layers")
            for param in self.models['C'].parameters():
                param.requires_grad = False
            for param in self.models['C'].fc.parameters():
                param.requires_grad = True
            '''
            self.models['C'] = self.models['C'].to(self.device)
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[20,35], gamma=0.1)
            self.epochs = 50

        #loading training data
        self.train_loader, self.val_loader, self.test_loader,self.steal_loader = \
            load_image(self.dataset, 1.0, self.train_pct, self.steal_pct, self.batch_size, self.imagesize, None, False)
        #self.train_pct
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            self.val_loader =  self.test_loader
        
        print(f"training size:{len(self.train_loader.dataset)}")
        #loading watermark training data
        print("Loading watermark training data...")
        if self.watermark_loader == None:  
            if self.dataset=='cifar10': #default：[8,9]
                self.train_loader2, _, _,_= \
            load_image(self.dataset, 1.0, 0.1,self.steal_pct, self.batch_size, 32, [8,9], True)
            elif self.dataset=='cifar100' and (self.model_arch=='resnet'or self.model_arch=='pre_resnet'):
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 0.1,self.steal_pct,self.batch_size, 32, [0,1,2], True)
            elif self.dataset=='caltech101':
                self.train_loader2, _, _,_= \
                load_image(self.dataset, 1.0, 1.0,self.steal_pct,self.batch_size, 32, [10], True)
            print("Number of wt images:", len(self.train_loader2.dataset))

            function_mapping = { 'Rotate90': image_rotate,'Grayscale':image_Grayscale,'ColorJitter': image_ColorJitter}
            print('Constructing watermark loader...')   
            self.wt_loader = loader_transform(self.train_loader2,function_mapping[self.wt_task],self.batch_size,100)
        else:
            self.wt_loader = self.watermark_loader
       
        print("Number of train images:", len(self.train_loader.dataset))
        print("Number of wt images:", len(self.wt_loader.dataset))

        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(self.optims['C'], iter_per_epoch * WARM)
        self.records['acc'] = 0.0
        self.acc1=[]
        self.acc2 =[]
         


    def train(self, epoch):
        self.models['C'].train()
        #adjust_learning_rate(init_lr=self.lr, optimizer=self.optims['C'], epoch=epoch, lradj=self.lradj)
        if self.delta == 0.0:
            for i,(x1, y1) in enumerate(self.train_loader):
                #data_loading_start = time.time()
                x1, y1= x1.to(self.device,non_blocking=True), y1.to(self.device,non_blocking=True)
                #data_loading_end = time.time()
                #print(f"Data loading time: {data_loading_end - data_loading_start:.5f} seconds")
                #if self.model_arch =='resnet' and x1.shape[1]==1:
                #    x1 = x1.repeat(1,3,1,1)
                #training_start = time.time()
                scores1 = self.models['C'](x1)
                loss = F.cross_entropy(scores1, y1)
                self.optims['C'].zero_grad()
                loss.backward()
                self.optims['C'].step()
                #training_end = time.time()
                #print(f"Model training time: {training_end - training_start:.5f} seconds")
                if epoch <= WARM:
                    self.warmup_scheduler.step()
                if i % 200 == 0:
                    self.logs['Loss'] = loss.item()
                    self.print_logs(epoch, i)
        else:
            for i,((x1, y1),(x2,y2)) in enumerate(zip(self.train_loader,cycle(self.wt_loader))):
                x1, y1= x1.to(self.device,non_blocking=True), y1.to(self.device,non_blocking=True)
                #if self.model_arch =='resnet' and x1.shape[1]==1:
                #    x1 = x1.repeat(1,3,1,1)
                scores1 = self.models['C'](x1)
                loss1 = F.cross_entropy(scores1, y1)

                x2, y2= x2.to(self.device), y2.to(self.device)
                #if x2.shape[1]==1 and self.model_arch =='resnet':
                #    x2 = x2.repeat(1,3,1,1)
                scores2 = self.models['C'](x2)
                scores2 = MAP(scores2,self.nz,self.partition,self.wt_bit)
                #scores2 = F.log_softmax(scores2/self.T, dim=1) #using soft logit as watermark prediction, increse robustness
                loss2 = F.cross_entropy(scores2, y2)

                loss = loss1+ self.delta * loss2
                #loss = loss1/(loss1.detach()+1e-6) + self.delta*(loss2/(loss2.detach()+1e-6))
                
                self.optims['C'].zero_grad()
                loss.backward()
                self.optims['C'].step()
                if epoch <= WARM:
                    self.warmup_scheduler.step()

                if i % 100 == 0:
                    if self.delta !=0:
                        self.logs['Loss1'] = loss1.item()
                        self.logs['Loss2'] = loss2.item()
                    self.logs['Loss'] = loss.item()

                    self.print_logs(epoch, i)
          
        if self.model_arch != 'mlp':
            self.scheduler.step()

        return loss.item()

    
    def eval(self, epoch):
        if epoch%2 ==0:
            acc1 = eval_task1(self.models['C'],self.test_loader,self.device,self.model_arch)
            self.acc1.append(acc1)
            is_best = False
            if acc1 >= self.records['acc']:
                is_best = True
                self.records['acc'] = acc1

            acc2,_,_ = eval_task2(self.models['C'],self.wt_loader,self.nz,self.partition,self.wt_bit,self.device,self.model_arch)
            self.acc2.append(acc2)
            #print(f'val acc: {acc1:.3f}')
            print(f'val acc: {acc1:.3f}, trigger acc: {acc2:.3f}')

        else:
            is_best = False
            
        return is_best

def loader_transform(loader,transform,batch_size,N):
        X=[]
        #Y =[]
        #Y_ori = []
        for i,(x, y) in enumerate(loader):
            print(i,batch_size*i,N)
            #if batch_size*i< N/2:

            if batch_size*i < N:
                
                X.append(x)
                print(y)
            else:
                break
        X = torch.cat(X,0)[:int(N/2)]
        X_rot90 = transform(X)
        Xw = torch.cat([X,X_rot90],0)
        #Xw = torch.cat(X,0)[:N]
        Yw = torch.zeros(N).long()
        Yw[int(N/4):int(N*3/4)]=1
        print(Xw.shape,Yw.shape)
        tensordata_rot = TensorDataset(Xw, Yw)
        loader_rot = DataLoader(tensordata_rot, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        return loader_rot
                

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

def get_data_dim(data_loader):
    X,y = next(iter(data_loader))
    nc= X.shape[1]
    return nc       
        
def MAP(scores,nz,partition,k):
    topk_indices = torch.topk(scores, k=k, dim=1)[1]
    masked_score2 = scores.clone()
    zeros_tensor = torch.zeros_like(scores)
    masked_score2.scatter_(1, topk_indices, zeros_tensor)
    score2_mean = torch.mean(masked_score2,dim=1,keepdim=True)
    #score2_std = torch.std(masked_score2,dim=1,keepdim=True)
    normalized_score2= (masked_score2 - score2_mean) 
    a = normalized_score2[:,partition]
    partition2 = [nz-1-a for a in partition]
    b = normalized_score2[:,partition2]
    scores = torch.stack((a.mean(dim=1),b.mean(dim=1)),dim=1)

    return scores
    

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



def eval_task2(net,loader_rot,nz,partition,wt_bit,device,model_arch):
    net.eval()
    correct = 0.0
    with torch.no_grad():
        SCORES = []
        Y=[]
        for x, y in loader_rot:
            if model_arch =='resnet' and x.shape[1]==1:
                x = x.repeat(1,3,1,1)
            x, y = x.to(device), y.to(device)
            N = len(x)
            #scores = trainer.models['C'](x)
            scores = net(x)
            Y.append(y)
            scores  = MAP(scores,nz,partition,wt_bit)

            soft_scores = F.softmax(scores,dim=1)
            SCORES.append(soft_scores)

            pred_y = torch.argmax(scores, dim=1)
            correct += torch.sum(pred_y == y).item()
    N = len(loader_rot.dataset)
    acc = correct / N
    #print('acc: {}'.format(acc))
    return round(acc,4),torch.cat(SCORES),torch.cat(Y)

def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.1 ** (epoch // lradj))
    print(f'current lr:{lr:.5f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
class multitask_Trainer(BaseTrainer, Confs0):

    def __init__(self,dataset,model_arch,watermark_loader,wt_task,wt_bit,delta,partition,manual_seed,device_ids=[0]):
        Confs0.__init__(self)
        self.dataset = dataset
        self.model_arch = model_arch
        self.wt_task = wt_task
        self.watermark_loader = watermark_loader   
        self.device_ids = device_ids

        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.partition = partition
        self.delta = delta
        self.wt_bit = wt_bit
        self.seed = manual_seed
        mt.utils.set_seed(manual_seed)
        print(f"delta:{self.delta}")
        if delta ==0.0:
            self.flag = f'mean-{self.dataset}-{model_arch}-delta{self.delta}-seed{manual_seed}'
        else:
            self.flag = f'mean-{self.dataset}-{model_arch}-{wt_task}-delta{self.delta}-seed{manual_seed}'
        #self.flag = f'clf-{self.dataset}-{model_arch}-seed{manual_seed}'
        BaseTrainer.__init__(self)
    
        
if __name__ == '__main__':
    trainer = multitask_Trainer()
    trainer.run(load_best=False, retrain=False)


    

