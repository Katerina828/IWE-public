import sys
sys.path.append('/workspace/mytoolbox')
import marveltoolbox as mt 
from src.clf2 import eval_task1,eval_task2,adjust_learning_rate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from models.resnet import ResNet18
from src.mlp import get_data_dim


class Confs0(mt.BaseConfs):
    def __init__(self):
        super().__init__()
        
    
    def get_dataset(self):
        self.dataset = 'mnist'
        self.nc = 1
        self.nz = 10
        self.epochs = 50
        self.input_size=600
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
    def __init__(self,target_trainer,shadow_dataloader):
        mt.BaseTrainer.__init__(self, self)
        self.train_loader = shadow_dataloader
        if self.model_arch =='resnet':
            self.models['C']  = ResNet18(num_classes=self.nz).to(self.device)
            #self.optims['C'] = torch.optim.Adam(self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))
            self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optims['C'], milestones=[50,75], gamma=0.1)
            self.epochs = 100
        elif self.model_arch == 'cnn': 
            self.models['C'] = mt.nn.dcgan.Enet32(self.nc, self.nz).to(self.device)
            if self.dataset=='cifar10':
                self.lr=0.01
                self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                #self.optims['C'] = torch.optim.Adam(self.models['C'].parameters(), lr=self.lr, betas=(0.5, 0.99))
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optims['C'], step_size=25, gamma=0.1, last_epoch=-1)
                self.epochs = 75
            elif self.dataset=='fmnist' or self.dataset=='mnist':
                self.lr=1e-4
                self.optims['C'] = torch.optim.Adam(self.models['C'].parameters(), lr=self.lr, betas=(0.5, 0.99))
                self.epochs = 60
        elif self.model_arch =='mlp':
            self.input_size= get_data_dim(self.train_loader)
            self.nz = 2
            self.hidden_size= 128
            self.models['C'] = mt.nn.mlp.MLP(input_size=self.input_size, output_size=self.nz, hidden_size=self.hidden_size ).to(self.device)
            self.optims['C'] = torch.optim.Adam(
            self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))
            self.epochs = 60
        
        self.test_loader = target_trainer.test_loader
        self.target_trainer = target_trainer
        #self.train_loader_rot = target_trainer.train_loader_rot
        self.records['acc'] = 0.0
        self.acc1=[]
        #self.acc2=[]

    def train(self,epoch):
        self.target_trainer.models['C'].eval()
        self.models['C'].train()
        #adjust_learning_rate(init_lr=self.lr, optimizer=self.optims['C'], epoch=epoch, lradj=20)

        logsoftmax = nn.LogSoftmax()
        for i, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            if self.model_arch =='resnet' and x.shape[1]==1:
                x = x.repeat(1,3,1,1)
            scores = self.models['C'](x)
            target_logits = self.target_trainer.models['C'](x) #teacher's prediction
            if self.hard_pred==True:
                y = torch.argmax(target_logits,dim=1)
                loss = F.cross_entropy(scores, y)
            else:
                y = F.softmax(target_logits,dim=1)
                loss = torch.mean(torch.sum(-y * F.log_softmax(scores, dim=1), dim=1))
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)
        if self.model_arch =='resnet':
            self.scheduler.step()
        return loss.item()

    def eval(self, epoch):
        
        acc1 = eval_task1(self.models['C'],self.test_loader,self.device,self.model_arch)
        #acc2,_,_ = eval_task2(self.models['C'],self.train_loader_rot,self.device,self.model_arch)
        is_best = False
        if acc1 >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc1
        print(f'val acc: {acc1:.3f}')
        self.acc1.append(acc1)
        #self.acc2.append(acc2)
        return is_best

class STrainer(BaseTrainer, Confs0):
    def __init__(self,dataset,model_arch,model_type,teacher_trainer,shadow_dataloader,hard_pred,manual_seed):
        Confs0.__init__(self)
        self.dataset = dataset
        self.model_arch = model_arch
        self.device_ids = teacher_trainer.device_ids
        self.nz = teacher_trainer.nz
        self.hard_pred = hard_pred
        if self.dataset=='mnist'or self.dataset=='fmnist':
            self.nc = 1
        elif self.dataset=='cifar10':
            self.nc = 3 
        self.device_ids = teacher_trainer.device_ids
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.seed = manual_seed
        mt.utils.seed.set_seed(manual_seed)
        
        self.flag = f'clf-{self.dataset}-{model_arch}-{model_type}-seed{manual_seed}'
        BaseTrainer.__init__(self,teacher_trainer,shadow_dataloader)
        
def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.1 ** (epoch // lradj))
    print(f'current lr:{lr:.5f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr