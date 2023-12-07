import sys
sys.path.append('/workspace/Research2/mytoolbox')
sys.path.append('/workspace/Research/MIA-image')
import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from nn.dcgan import Enet32
from torch.utils.data import TensorDataset,DataLoader
from src.clf2 import eval_task1,eval_task2
from models import *

class Confs0(mt.BaseConfs):
    def __init__(self):
        super().__init__()
        
    
    def get_dataset(self):
        self.dataset = 'mnist'
        self.nc = 1
        self.nz = 10
        self.epochs = 50
        self.batch_size= 100
        self.lr=0.01

    def get_flag(self):
        self.flag = 'dcgan-{}-clf-steal'.format(self.dataset)
    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

class BaseTrainer(mt.BaseTrainer):
    def __init__(self,teacher_trainer):
        mt.BaseTrainer.__init__(self, self)
        '''
        self.models['C']  = ResNet18(num_classes=self.nz).to(self.device)
        #self.models['C'] = mt.nn.dcgan.Enet32(self.nc, self.nz).to(self.device)
        self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        #self.optims['C'] = torch.optim.Adam(
        #    self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))
        '''
        if self.model_arch =='resnet':
            self.models['C']  = ResNet18(num_classes=self.nz).to(self.device)
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optims['C'], step_size=2, gamma=0.9, last_epoch=-1)
            self.epochs = 60
        elif self.model_arch == 'cnn':  
            
            self.models['C'] = mt.nn.dcgan.Enet32(self.nc, self.nz).to(self.device)
            if self.dataset=='cifar10':
                self.lr=0.01
                self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

                #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optims['C'], step_size=2, gamma=0.9, last_epoch=-1)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optims['C'], step_size=25, gamma=0.1, last_epoch=-1)
                self.epochs = 75
            elif self.dataset=='fmnist':
                self.lr=1e-4
                self.optims['C'] = torch.optim.Adam(self.models['C'].parameters(), lr=self.lr, betas=(0.5, 0.99))
                self.epochs = 60
            
        self.records['acc'] = 0.0
        self.acc1=[]
        self.acc2=[]
        
        self.val_loader = teacher_trainer.test_loader
        self.teacher_model = teacher_trainer.models['C']
        self.train_loader_rot = teacher_trainer.train_loader_rot

    def train(self,epoch):
        self.teacher_model.eval()
        self.models['C'].train()
        #adjust_learning_rate(init_lr=self.lr, optimizer=self.optims['C'], epoch=epoch, lradj=20)
        logsoftmax = nn.LogSoftmax()
        for i, (x, y) in enumerate(self.train_loader):
            if self.model_arch =='resnet' and x.shape[1]==1:
                x = x.repeat(1,3,1,1)
            x,y = x.to(self.device),y.to(self.device)
            scores = self.models['C'](x)
            teacher_outputs = self.teacher_model(x)
            loss = self.loss_fn_kd(scores, y, teacher_outputs)
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)
        if self.dataset =='cifar10':
            self.scheduler.step()
        return loss.item()

    def loss_fn_kd(self,outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        alpha= 0.5
        T = 1.5
        """
        criterion_kl = nn.KLDivLoss(reduction = "batchmean")
        
        KD_loss = criterion_kl(F.log_softmax(outputs/self.T, dim=1),
                                F.softmax(teacher_outputs/self.T, dim=1)) * (self.alpha * self.T * self.T) + \
                F.cross_entropy(outputs, labels) * (1. - self.alpha)

        return KD_loss

    def eval(self, epoch):
        
        acc1 = eval_task1(self.models['C'],self.val_loader,self.device,self.model_arch)
        acc2,_,_ = eval_task2(self.models['C'],self.train_loader_rot,self.device,self.model_arch)
        is_best = False
        if acc1 >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc1
        print(f'val acc: {acc1:.3f}, trigger acc: {acc2:.3f}')
        self.acc1.append(acc1)
        self.acc2.append(acc2)
        return is_best

class KDTrainer(BaseTrainer, Confs0):
    def __init__(self,dataset,model_arch,model_type,teacher_trainer,train_sample_loader,manual_seed,alpha=0.5,T=1.5):
        Confs0.__init__(self)
        self.seed = manual_seed
        mt.utils.seed.set_seed(manual_seed)
        self.dataset = dataset
        self.model_arch = model_arch
        if self.dataset=='mnist':
            self.nc = 1
        elif self.dataset=='cifar10':
            self.nc = 3
            self.epochs = 60
        self.alpha = alpha
        self.T = T
        self.train_loader = train_sample_loader
        self.device_ids = teacher_trainer.device_ids
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.nz = 10
        
        self.flag = f'clf-{self.dataset}-{model_arch}-{model_type}-seed{manual_seed}'
        BaseTrainer.__init__(self,teacher_trainer)
        
def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.9 ** (epoch // lradj))
    print(f'current lr:{lr:.5f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr