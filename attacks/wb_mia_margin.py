import sys
sys.path.append('/workspace/Research2/mytoolbox')
sys.path.append('/workspace/Research/MIA-image')
import marveltoolbox as mt 
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset,DataLoader
import numpy as np



def get_margin(net, inputs,labels):
    cw= mt.attacks.cw.CWAttack
    best_adv_x, labels= cw.attack_batch(net, inputs, labels)
    D,D = cw.D_loss(inputs, best_adv_x)
    return D
    