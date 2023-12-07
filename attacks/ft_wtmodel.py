import sys
sys.path.append('/workspace/Research2/mytoolbox')
sys.path.append('/workspace/Research/MIA-image')
import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models
from src.rdp_accountant import compute_rdp, get_privacy_spent
from hat_datasets.load_data import load_image
from nn.wideresnet import WideResNet
from src.clf2 import eval_task1,eval_task2

class Confs0(mt.BaseConfs):
    def __init__(self):
        super().__init__()

    def get_dataset(self):
        self.dataset = 'cifar2'
        self.nc = 3
        self.nz = 2
        self.epochs = 50
        self.batch_size= 100
        self.sigma = 0
        self.steps = 0
        self.lr = 0.05

    def get_flag(self):
        self.flag = 'dcgan-{}-clf'.format(self.dataset)
    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        

def freeze_hidden_layers(net):
    for param in net.parameters():
        param.requires_grad = False 
    
    for index,(name, param) in enumerate(net.named_parameters()):
        if index > (len(list(net.named_parameters()))-3):
            param.requires_grad = True
    return net


class BaseTrainer(mt.BaseTrainer):
    def __init__(self,Teacher_trainer,dataloader):
        mt.BaseTrainer.__init__(self, self)
        self.model_arch = Teacher_trainer.model_arch
        self.models['C']= Teacher_trainer.models['C'].to(self.device)

        
        self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.train_loader=  dataloader
        self.val_loader=  Teacher_trainer.test_loader
        self.test_loader = Teacher_trainer.test_loader
        self.trigger_loader = Teacher_trainer.train_loader_rot


        self.records['acc'] = 0.0
        self.acc1=[]
        self.acc2 =[]

    def train(self, epoch):
        adjust_learning_rate(init_lr=self.lr, optimizer=self.optims['C'], epoch=epoch, lradj=1)
        self.models['C'].train()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            if self.model_arch =='resnet' and x.shape[1]==1:
                x = x.repeat(1,3,1,1)
            scores = self.models['C'](x)
            loss = F.cross_entropy(scores, y)
            self.optims['C'].zero_grad()
            loss.backward()

            self.optims['C'].step()
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)
        
        return loss.item()
                
    def eval(self, epoch):
        
        acc1 = eval_task1(self.models['C'],self.val_loader,self.device,self.model_arch)
        acc2,_,_ = eval_task2(self.models['C'],self.trigger_loader,self.device,self.model_arch)
        is_best = False
        if acc1 >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc1
        print(f'val acc: {acc1:.3f}, trigger acc: {acc2:.3f}')
        self.acc1.append(acc1)
        self.acc2.append(acc2)
        return is_best

def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.9 ** (epoch // lradj))
    print(f'current lr:{lr:.5f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

        
class Finetune(BaseTrainer, Confs0):##cifar100
    def __init__(self,dataset,model_type,teacher_trainer,dataloader,manual_seed):
        Confs0.__init__(self)
        self.dataset = dataset
        if self.dataset=='mnist':
            self.nc = 1
        elif self.dataset=='cifar10':
            self.nc = 3
            self.epochs = 60
        self.device_ids = teacher_trainer.device_ids
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.seed = manual_seed
        mt.utils.seed.set_seed(manual_seed)
        self.flag = f'clf-{self.dataset}-{model_type}-seed{manual_seed}'
        BaseTrainer.__init__(self,teacher_trainer,dataloader)



    
        
if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(load_best=False, retrain=False)


    

