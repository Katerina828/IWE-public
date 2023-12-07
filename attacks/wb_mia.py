import sys
sys.path.append('/workspace/Research2/mytoolbox')
sys.path.append('/workspace/Research/MIA-image')
import marveltoolbox as mt 
import torch
import torch.nn.functional as F
from src.grad_clf import Trainer
from src.utils import compute_acc
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
from sklearn.neural_network import MLPClassifier


def get_gradient(trainer,dataloader,conv_flag=True):
    net = trainer.models['C'].cpu()
    net.eval()
    for param in net.parameters():
        param.requires_grad = False 
    #for param in net.fc3.parameters():
    #    param.requires_grad = True
    for index,(name, param) in enumerate(net.named_parameters()):
        if index == (len(list(net.named_parameters()))-2):
            param.requires_grad = True
    
    GD = []
    Target = []
    for i,(x, y) in enumerate(dataloader):
        #x, y = x.to(trainer.device), y.to(trainer.device)
        Target.append(y)
        N=len(x)
        scores = net(x).float()
        loss = F.cross_entropy(scores, y)
        net.zero_grad()
        loss.backward()
        
        if conv_flag == True:
            gradient = net.output[3].weight.grad.clone()   #conv
        else:
            gradient = net.fc3.weight.grad.clone()   #fc3
        gradient = torch.squeeze(gradient)
        GD.append(gradient)
        if ((i+2)*N)>10000:
            break
    GD = torch.stack(GD,dim=0)
    Target= torch.stack(Target,dim=0)
    return GD,Target

def get_wb_input(mem_dataloader,nmem_dataloader,trainer,conv_flag):

    mem_GD,mem_Target = get_gradient(trainer, mem_dataloader,conv_flag=conv_flag)
    nmem_GD,nmem_Target = get_gradient(trainer, nmem_dataloader,conv_flag=conv_flag)
    
    mem_GD_norm = torch.flatten(mem_GD, start_dim=1).norm(2,dim=1)
    nmem_GD_norm = torch.flatten(nmem_GD, start_dim=1).norm(2,dim=1)


    GD = torch.cat([mem_GD,nmem_GD], dim=0).cpu()
    GD_norm = torch.cat([mem_GD_norm,nmem_GD_norm], dim=0).cpu()
    Targets = torch.cat([mem_Target,nmem_Target], dim=0).cpu()
    
    mem_label = torch.ones(len(mem_GD))
    n_mem_label = torch.zeros_like(mem_label)
    label = torch.cat([mem_label,n_mem_label],dim=0).detach().cpu()      
    return GD,GD_norm,Targets,label


def wb_mia(train_loader, valid_loader,shadow_loader,test_loader,shadow_trainer,trainer,conv_flag,attack_mode=1):

    print(len(train_loader),len(valid_loader),len(shadow_loader),len(test_loader))

    #perparing attack trainloader

    GD,GD_norm_shadow,Targets_shadow,label_shadow = get_wb_input(valid_loader,shadow_loader,shadow_trainer,conv_flag=conv_flag)

    if attack_mode==1:

        GD = GD.unsqueeze(1).permute(0,1,3,2)
        GD_mean=torch.mean(GD,dim=(0,2,3),keepdim=True)
        GD_std = torch.std(GD,dim=(0,2,3),keepdim=True)
        nomalized_GD = (GD-GD_mean)/GD_std
        
        attack_train = nomalized_GD
        train_set = TensorDataset(attack_train, label_shadow.long())
        attack_train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1)

        # perparing attack testloader
    GD,GD_norm_victim,Targets_victim,label_victim = get_wb_input(train_loader,test_loader,trainer,conv_flag=conv_flag)
    if attack_mode==1:
        GD = GD.unsqueeze(1).permute(0,1,3,2)
        nomalized_GD = (GD-GD_mean)/GD_std
        
        attack_test = nomalized_GD
        
        test_set = TensorDataset(attack_test, label_victim.long())
        attack_test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=1)

        print(GD_norm_victim[:10000].mean(),GD_norm_victim[10000:].mean())
        
        attack_trainer= Trainer('Gradient')
        attack_trainer.train_loader = attack_train_loader
        attack_trainer.val_loader = attack_test_loader
        attack_trainer.test_loader = attack_train_loader
        
        # run and evalute the MIA effect
        attack_trainer.run(load_best=False, retrain=True)

        compute_acc(attack_trainer, attack_train_loader,attack_test_loader)
    
    if attack_mode==2:
        GD_norm_mean = torch.mean(GD_norm_shadow,dim=(0),keepdim=True)
        GD_norm_std = torch.std(GD_norm_shadow,dim=(0),keepdim=True)
        normazelied_GD_norm_shadow = (GD_norm_shadow-GD_norm_mean)/GD_norm_std

        normazelied_GD_norm_victim = (GD_norm_victim-GD_norm_mean)/GD_norm_std

        onehottargets_shadow = F.one_hot(torch.squeeze(Targets_shadow), num_classes=100).float()
        onehottargets_victim = F.one_hot(torch.squeeze(Targets_victim), num_classes=100).float()

        X_train = np.concatenate([normazelied_GD_norm_shadow.view(-1,1).numpy(),onehottargets_shadow.numpy()],axis=1)
        y_train = label_shadow.numpy()
        print(X_train.shape,y_train.shape)


        X_test = np.concatenate([normazelied_GD_norm_victim.view(-1,1).numpy(),onehottargets_victim.numpy()],axis=1)
        y_test = label_victim.numpy()
        print(X_test.shape,y_test.shape)

        classifier = MLPClassifier(hidden_layer_sizes=(64, ),max_iter = 200)
        classifier.fit(X_train, y_train)
        y_score = classifier.predict(X_train)
        acc=   np.sum(y_score == y_train)/(len(X_train))
        print("train acc: %.4f"%(acc))
        y_score = classifier.predict(X_test)
        acc=   np.sum(y_score == y_test)/(len(X_train))
        print("test acc: %.4f"%(acc))

#   


