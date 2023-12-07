import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import random
from random import shuffle
import pandas as pd
from hat_datasets.transformer import BaseTransformer

def load_purchase(
    n_train_examples: int = 50000,n_valid_examples: int = 30000,n_shadow_nonmember_examples:int = 10000,n_test_examples: int = 10000,  batch_size: int = 100):
    torch.manual_seed(0)
    df =  pd.read_csv('hat_datasets/rawdata/purchase100.txt', header = None)
    
    
    target = torch.tensor(df[600].values).long()
    features = df.drop(columns=[600]).values.astype(np.float32)

    features_tensor = torch.tensor(features)
    train_valid_set = TensorDataset(features_tensor, target)
    

    #n_train_examples = int(train_pct*len(train_valid_set))
    #n_valid_examples = int(valid_pct*len(train_valid_set))
    #n_test_examples = int(test_pct*len(train_valid_set))
    train_set, valid_set, s_nonmember_set,test_set,_ = torch.utils.data.random_split(
            train_valid_set,
            lengths=[
                n_train_examples,
                n_valid_examples,
                n_shadow_nonmember_examples,
                n_test_examples, 
                len(train_valid_set) - n_train_examples-n_valid_examples-n_shadow_nonmember_examples-n_test_examples,
            ],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=1)
    shadow_loader =DataLoader(s_nonmember_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)
    #remained_loader = DataLoader(remained_set, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader, valid_loader,test_loader