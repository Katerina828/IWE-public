import sys
sys.path.append('/workspace/Research2/mytoolbox')
sys.path.append('/workspace/Research/MIA-image')
import marveltoolbox as mt 
import torch
import torch.nn.functional as F
from src.clf2 import eval_task1
import argparse

#from src.wt_train_mlp import eval_task2

def adjust_learning_rate(init_lr, optimizer, epoch, lradj, ratio = 0.1, period = 99999):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (ratio ** (epoch // lradj))
    print(f'current lr:{round(lr,8)}' )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        

    

class Confs0(mt.BaseConfs):
    def __init__(self):
        super().__init__()

    def get_dataset(self):
        self.dataset = 'cifar2'
        self.nc = 3
        self.nz = 2
        self.batch_size= 50
        self.sigma = 0
        self.steps = 0
        self.EWC_samples = 10000
        self.EWC_coef = 0



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
        self.wt_loader = Teacher_trainer.wt_loader
        self.nz = Teacher_trainer.nz
        if hasattr(Teacher_trainer,"partition"):
            self.partition = Teacher_trainer.partition
            self.wt_flag = 'mean'
            self.wt_bit = Teacher_trainer.wt_bit
        '''
        if hasattr(Teacher_trainer,"wt_bit"):
            print("Use mlp to embed watermarks")
            self.wt_bit = Teacher_trainer.wt_bit
            self.models['L'] = Teacher_trainer.models['L']
            self.wt_flag = 'mlp'
        '''
        self.models['C']= Teacher_trainer.models['C'].to(self.device)
        self.optims['C'] = torch.optim.SGD(self.models['C'].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.train_loader =  dataloader
        self.val_loader=  Teacher_trainer.test_loader
        self.test_loader = Teacher_trainer.test_loader
        #self.trigger_loader = Teacher_trainer.train_loader_rot

        if self.EWC_coef>0:
            self.Fisher,self.init_params =get_fisher(self.models['C'],self.train_loader,self.EWC_samples,self.device)

        self.records['acc'] = 0.0
        self.acc1=[]
        self.acc2 =[]

    def train(self, epoch):
        adjust_learning_rate(init_lr=self.lr, optimizer=self.optims['C'], epoch=epoch, lradj=self.lradj,ratio = self.ratio)
        self.models['C'].train()
        for i, (x, y) in enumerate(self.train_loader):
    
            x, y = x.to(self.device), y.to(self.device)
            if self.model_arch =='resnet' and x.shape[1]==1:
                x = x.repeat(1,3,1,1)
            scores = self.models['C'](x)
            loss = F.cross_entropy(scores, y)

            if self.EWC_coef > 0:
                for param, fisher, init_param in zip(self.models['C'].parameters(), self.Fisher, self.init_params):
                    loss = loss + (0.5 * self.EWC_coef * fisher.clamp(max = 1. / self.optims['C'].param_groups[0]['lr'] / self.EWC_coef) * ((param - init_param)**2)).sum()
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()

            if i %100  == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)
                #self.eval(epoch)
                #self.models['C'].train()
            if self.epochs==1:
                if i >= self.iteration:
                    print("end training.")
                    break
                
        return loss.item()
                
    def eval(self, epoch):
        self.models['C'].eval()
        acc1 = eval_task1(self.models['C'],self.val_loader,self.device,self.model_arch)
        #acc2,_,_ = eval_task2(self.models['C'],self.wt_loader,self.partition,self.device,self.model_arch)
        if self.wt_flag =='mean':
            from src.wt_train_mean import eval_task2
            acc2,_,_ = eval_task2(self.models['C'],self.wt_loader,self.nz,self.partition,self.wt_bit,self.device,self.model_arch)
        elif self.wt_flag =='mlp':
            from src.wt_train_mlp import eval_task2
            acc2,_,_ = eval_task2(self.models['C'],self.models['L'],self.wt_loader,self.wt_bit,self.device,self.model_arch)
        is_best = False
        if acc1 >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc1
        print(f'val acc: {acc1:.4f}, trigger acc: {acc2:.4f}')
        self.acc1.append(acc1)
        self.acc2.append(acc2)
        return is_best


def IsInside(x, Y):
    for y in Y:
        if x is y:
            return True
    return False
        
class Finetune(BaseTrainer, Confs0):##cifar100
    def __init__(self,dataset,model_type,teacher_trainer,dataloader,manual_seed,lr,lradj,ratio,epochs,iteration):
        Confs0.__init__(self)
        self.dataset = dataset
        if self.dataset=='mnist':
            self.nc = 1
        elif self.dataset=='cifar10':
            self.nc = 3
        steal_pct = len(dataloader.dataset)/len(teacher_trainer.train_loader.dataset)
        print("finetune data pct:", round(steal_pct,2))
        #self.EWC_coef = EWC_coef #if EWC_coef,then open EWC; else, do not use EWC
        self.lr= lr
        self.lradj = lradj
        self.ratio = ratio
        self.epochs = epochs
        self.iteration = iteration
        self.nz = teacher_trainer.nz
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


def get_fisher(net,trainloader,EWC_samples,device):
    net.eval()
    grad_sum = [param.new_zeros(param.size()) for param in net.parameters()]
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)#this line is not the optimizer used for actual training!

    sample_cnt = 0
    while True:
        for inputs, targets in trainloader:
            if sample_cnt >= EWC_samples:
                continue

            inputs, targets = inputs.to(device), targets.to(device)
            prob = F.softmax(net(inputs), dim=1)
            lbls = torch.multinomial(prob, 1).to(device)
            log_prob = torch.log(prob)
        
            for i in range(inputs.size(0)):
                optimizer.zero_grad()
                log_prob[i][lbls[i]].backward(retain_graph=True)
                with torch.no_grad():
                    grad_sum = [g + (param.grad.data.detach()**2) for g, param in zip(grad_sum, net.parameters())]

            sample_cnt += inputs.size(0)
            print ("Approximating Fisher: %.3f"%(float(sample_cnt) / EWC_samples))
        if sample_cnt >= EWC_samples:
            break
    
    Fisher = [g / sample_cnt for g in grad_sum]


    _fmax = 0
    _fmin = 1e9
    _fmean = 0.
    for g in Fisher:
        _fmax = max(_fmax, g.max())
        _fmin = min(_fmin, g.min())
        _fmean += g.mean()
    print ("[max: %.3f] [min: %.3f] [avg: %.3f]"%(_fmax, _fmin, _fmean / len(Fisher)))

    Fisher = [g / _fmax for g in Fisher]

    init_params = [param.data.clone().detach() for param in net.parameters()]
    
    return Fisher,init_params
    
        
if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(load_best=False, retrain=False)


    

