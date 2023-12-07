import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from attacks.adv_attacks import rand_steps
from attacks.adv_attacks import *



class Args():
    def __init__(self):
        self.dataset='cifar10'
        self.batch_size=50
        self.num_classes = 10
        self.device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")
        self.num_classes = 10
        self.gap = 0.001
        self.k = 100
        self.num_iter = 500
        self.alpha_l_1=1.0  
        self.alpha_l_2=0.01
        self.alpha_l_inf =0.001 


def get_random_label_only(loader, trainer, num_images = 1000):
    print("Getting random attacks")
    args= Args()
    args.device = trainer.device
    model = trainer.models['C'].to(args.device)
    batch_size = 50
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]

    for i,batch in enumerate(loader):

        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(10): #5 random starts
                X,y = batch[0].to(args.device), batch[1].to(args.device) 
                args.distance = distance
                # args.lamb = 0.0001
                preds = model(X)
                targets = None
                delta = rand_steps(model, X, y, args, target = targets)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d



def get_mingd_vulnerability(loader, trainer, num_images = 1000):
    args= Args()
    args.device = trainer.device
    model = trainer.models['C'].to(args.device)
    max_iter = num_images/args.batch_size
    lp_dist = [[],[],[]]

    for i,batch in enumerate(loader):

        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(args.num_classes):
                X,y = batch[0].to(args.device), batch[1].to(args.device) 
                args.distance = distance
                # args.lamb = 0.0001
                delta = mingd(model, X, y, args, target = y*0 + target_i)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d