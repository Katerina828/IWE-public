import sys
sys.path.append('/workspace/Research2/mytoolbox')
sys.path.append('/workspace/Research/MIA-image')
import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.mlp_attackmodel import Trainer
from src.utils import compute_acc
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

## features[:100] p_scores
# features[100:200] one-hot encode labels  
def get_bb_input(mem_dataloader,nmem_dataloader,trainer,num_classes):

    mem_Posteriors,mem_Target,mem_onehot_target = get_posteriors(trainer, mem_dataloader,num_classes)
    nmem_Posteriors, nmem_Target,nmem_onehot_target = get_posteriors(trainer, nmem_dataloader,num_classes)
    N = min(len(mem_Posteriors),len(nmem_Posteriors))
    
    Posteriors = torch.cat([mem_Posteriors[0:N],nmem_Posteriors[0:N]], dim=0).cpu()
    Targets = torch.cat([mem_Target[0:N],nmem_Target[0:N]], dim=0).cpu()
    OneHotTargets = torch.cat([mem_onehot_target[0:N],nmem_onehot_target[0:N]], dim=0).cpu()

    mem_label = torch.ones(N)
    n_mem_label = torch.zeros_like(mem_label)
    labels = torch.cat([mem_label,n_mem_label],dim=0).cpu()  
    
    return Posteriors,Targets,OneHotTargets,labels

def get_posteriors(trainer, dataloader,num_classes):
    net = trainer.models['C']
    net.eval()
    Posteriors = []
    Target = []
    OneHotTarget = []
    with torch.no_grad():
        for i,(x, y) in enumerate(dataloader):
            x, y = x.to(trainer.device), y.to(trainer.device)
            N=len(x)
            
            one_hot_target = F.one_hot(y, num_classes=num_classes).float()
            Target.append(y)
            OneHotTarget.append(one_hot_target)
            scores = net(x).float()
            
            p_scores = F.softmax(scores,dim=1)

            Posteriors.append(p_scores)
            
            if ((i+2)*N)>10000:
                break
        Posteriors = torch.cat(Posteriors,dim=0)
        Target= torch.cat(Target,dim=0)
        OneHotTarget  = torch.cat(OneHotTarget,dim=0)
    return Posteriors,Target,OneHotTarget

def get_logits(p_scores,y_target):
    COUNT = p_scores.shape[0]
    print(COUNT)
    y_true = p_scores[torch.arange(COUNT),y_target]

    p_scores[torch.arange(COUNT),y_target] =0
    y_wrong = torch.sum(p_scores,axis=1)

    logit = torch.log(y_true+1e-45)-torch.log(y_wrong+1e-45)
    return logit


def get_max_posterior(p_scores):
    pred_y = torch.argmax(p_scores, dim=1)
    max_posterior =  p_scores[torch.arange(len(p_scores)),pred_y].cpu().numpy()
    return max_posterior.reshape(-1,1)

def get_entropy(p_scores):
    entropy = (-p_scores.log()*p_scores).sum(dim=1).cpu().numpy().astype('float64')
    entropy[np.isnan(entropy)] = 0
    return entropy.reshape(-1,1)

def get_loss(p_scores,y_target):
    log_soft_out = torch.log(p_scores)
    loss = F.nll_loss(log_soft_out, y_target.long(),reduction ='none').cpu().numpy()
    return loss.reshape(-1,1)


# support attack mode
# 1. full posteriors + target label
# 2. maximal posteriors + target label
# 3. loss 
# 4. entropy

def bb_mia(train_loader, valid_loader,shadow_loader,test_loader,
            shadow_trainer,trainer,attack_mode=1,sklearn=True, nn=False):
    
    Posteriors,Targets,OneHotTarget,labels = get_bb_input(valid_loader,shadow_loader,shadow_trainer)

    if attack_mode == 1:
        X_train = np.concatenate([Posteriors.numpy(),OneHotTarget.numpy()],axis=1)
    elif attack_mode == 2:
        max_posterior = get_max_posterior(Posteriors)
        X_train = np.concatenate([max_posterior,OneHotTarget.numpy()],axis=1)
    elif attack_mode == 3:
        X_train = get_entropy(Posteriors)
    elif attack_mode == 4:
        X_train = get_loss(Posteriors,Targets)
    y_train = labels.numpy()
    print("Training data shape:",X_train.shape,y_train.shape)

    Posteriors,Targets,OneHotTarget,labels = get_bb_input(train_loader,test_loader,trainer)

    if attack_mode == 1:
        X_test = np.concatenate([Posteriors.numpy(),OneHotTarget.numpy()],axis=1)
    elif attack_mode == 2:
        max_posterior = get_max_posterior(Posteriors)
        X_test = np.concatenate([max_posterior,OneHotTarget.numpy()],axis=1)
    elif attack_mode == 3:
        X_test = get_entropy(Posteriors)
    elif attack_mode == 4:
        X_test = get_loss(Posteriors,Targets)

    y_test = labels.numpy()
    print("Testing data shape:",X_test.shape,y_test.shape)

    if sklearn== True:
        print('Using sklearn mlp classifier')
        classifier = MLPClassifier(hidden_layer_sizes=(64, ),max_iter = 200)
        classifier.fit(X_train, y_train)
        y_score = classifier.predict(X_train)
        acc=   np.sum(y_score == y_train)/(len(X_train))
        print("train acc: %.4f"%(acc))
        
        y_score = classifier.predict(X_test)
        acc=   np.sum(y_score == y_test)/(len(X_train))
        print("test acc: %.4f"%(acc))

        auroc = metrics.roc_auc_score(y_test, y_score) 
        print("MIA Attack auroc: %.4f"%(auroc))
    
    if nn==True:
        attack_trainer= Trainer('attackmodel')
        
        train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
        attack_train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1)
        
        test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
        attack_test_loader = DataLoader(test_set, batch_size=256, shuffle=True, num_workers=1)

        attack_trainer.train_loader = attack_train_loader
        attack_trainer.val_loader = attack_test_loader
        #attack_trainer.test_loader = attack_train_loader

        attack_trainer.run(load_best=False, retrain=True)
        
        compute_acc(attack_trainer, attack_train_loader,attack_test_loader)

def bb_mia_threshold(train_loader,test_loader,trainer,num_classes):
    print('begin attack')
    Posteriors,Targets,_,labels = get_bb_input(train_loader,test_loader,trainer,num_classes)
    print(labels.shape)
    max_posterior = get_max_posterior(Posteriors)
    entropy= get_entropy(Posteriors)
    print("entropy shape,",entropy.shape)
    loss = get_loss(Posteriors,Targets)
    logit = get_logits(Posteriors,Targets)

    df= pd.DataFrame(max_posterior,columns=['max_posterior'])
    df['label'] = labels
    df['loss'] = loss
    df['entropy'] = entropy
    df['logit'] = logit

    df.fillna(value = {'entropy': 0},inplace=True)

    auroc = roc_auc_score(df['label'], df['max_posterior']) 
    print("AUROC on maximum posteriors:", round(auroc,4))
    auroc = roc_auc_score(df['label'], -df['loss']) 
    print("AUROC on loss:", round(auroc,4))
    auroc = roc_auc_score(df['label'], -df['entropy']) 
    print("AUROC on entropy:", round(auroc,4))
    auroc = roc_auc_score(df['label'], df['logit']) 
    print("AUROC on logit:", round(auroc,4))
    return df

def plot_df(df):
    sns.set(style="whitegrid", color_codes=True,font_scale=1.5)
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,sharey=True,figsize=(12, 4))
    plt.subplots_adjust(wspace =0.05, hspace =0)
    g1 = sns.histplot(data = df,x='max_posterior',hue="label",log_scale=(False,True),bins=30,ax=ax1)
    g2 = sns.histplot(data = df,x='loss',hue="label",log_scale=(False,True),bins=30,ax=ax2)
    g3 = sns.histplot(data = df,x='entropy',hue="label",log_scale=(False,True),bins=30,ax=ax3)
    if 'margin' in df.columns:
        g4 = sns.histplot(data = df,x='margin',hue="label",log_scale=(False,True),bins=30,ax=ax4)
    #g4 = sns.histplot(data = df,x='gd_norm',hue="label",log_scale=(False,True),bins=30,ax=ax4)

    #plt.savefig("./figure/cifar100_MIA.png",bbox_inches='tight',dpi=300,pad_inches=0.0)