import sys
sys.path.append('/workspace/Research2/mytoolbox')
sys.path.append('/workspace/Research/MIA-image')
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import marveltoolbox as mt 
from src.clf2 import eval_task1
from attacks.finetune import Confs0
'''
Kang Liu, Brendan Dolan-Gavitt, and Siddhart Garg. 
Fine-Pruning: Defending against backdooring attacks on deep neural networks.
RAID 2018

'''
activations = []

def get_activations_hook(module, input, output):
        #output = module(input)
        activations.append(output)

#prune filters of conv layer
def prune_lastconvlayer(trainer,dataloader,p):
    trainer.models['C'].eval()
    
    hook = trainer.models['C'].layer4[-1].conv2.register_forward_hook(get_activations_hook)
    with torch.no_grad():
        for data,_ in dataloader:
            data = data.to(trainer.device)
            output = trainer.models['C'](data)
            activations.append(activations[0])
        activations_tensor = torch.cat(activations,dim=0)
        activations_mean = torch.mean(activations_tensor,dim=(0,2,3))


    #pruning settings
    prune_percent = p
    conv2_num = trainer.models['C'].layer4[-1].conv2.weight.shape[0]
    num_filters_to_prune = int(prune_percent * conv2_num)
    print(f"Total neuron {conv2_num},number of filters to prune:{num_filters_to_prune}")

    #Sort the filters based on their activations in ascending order
    _, sorted_indices  = torch.sort(activations_mean)

    # Prune the specified number of flters with the lowest activation
    for i in range(num_filters_to_prune):
        filter_idx = sorted_indices[i]
        trainer.models['C'].layer4[-1].conv2.weight.data[filter_idx,:,:,:] = 0.

    hook.remove()
    return trainer

#have not experiment with it
def prune_weight_inallconvlayer(trainer,dataloader,p):
    trainer.models['C'].eval()
    total = 0
    for m in trainer.models['C'].modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()
    conv_weights = torch.zeros(total)
    index = 0
    for m in trainer.models['C'].modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * p)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(trainer.models['C'].modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))


class BaseTrainer(mt.BaseTrainer):
    def __init__(self,Teacher_trainer,dataloader):
        mt.BaseTrainer.__init__(self, self)
        self.model_arch = Teacher_trainer.model_arch
        self.models['C']= Teacher_trainer.models['C'].to(self.device)
        self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.train_loader =  dataloader
        self.val_loader=  Teacher_trainer.test_loader
        self.test_loader = Teacher_trainer.test_loader
        #self.trigger_loader = Teacher_trainer.train_loader_rot
        self.records['acc'] = 0.0
        self.acc1=[]
        self.acc2 =[]

    def train(self, epoch):
        
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
        #acc1 = eval_task1(self.models['C'],self.val_loader,self.device,self.model_arch)
        acc1 = eval_task1(self.models['C'],self.val_loader,self.device,self.model_arch)
        is_best = False
        if acc1 >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc1
        print(f'val acc: {acc1:.4f}')
        self.acc1.append(acc1)
        return is_best



       
class FTonFP(BaseTrainer, Confs0):##cifar100
    def __init__(self,dataset,model_type,teacher_trainer,dataloader,manual_seed):
        Confs0.__init__(self)
        self.dataset = dataset
        if self.dataset=='mnist':
            self.nc = 1
        elif self.dataset=='cifar10':
            self.nc = 3
        steal_pct = len(dataloader.dataset)/len(teacher_trainer.train_loader.dataset)
        print("finetune data pct:", round(steal_pct,2))
        #self.EWC_coef = EWC_coef #if EWC_coef,then open EWC; else, do not use EWC
        self.nz = teacher_trainer.nz
        self.lr = 0.001
        self.device_ids = teacher_trainer.device_ids
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.seed = manual_seed
        mt.utils.seed.set_seed(manual_seed)
        if self.dataset =='caltech101':
            self.flag = f'clf-{self.dataset}-{model_type}-{10}-seed{manual_seed}'
        else:
            self.flag = f'clf-{self.dataset}-{model_type}-stealpct{round(steal_pct,2)}-{10}-seed{manual_seed}'
        BaseTrainer.__init__(self,teacher_trainer,dataloader)

