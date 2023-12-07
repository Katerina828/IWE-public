import numpy as np
import pandas as pd
import torch
from os import path
from hat_datasets.utils import get_data_info
from hat_datasets.transformer import BaseTransformer
from torch.utils.data import TensorDataset,DataLoader




class LawschoolDataset():
    
    def __init__(self):
        data_dir = 'hat_datasets/rawdata'
        data_file = path.join(data_dir, 'lawschs1_1.dta')
        df = pd.read_stata(data_file)

        self.data, cat_vars = self.lawsch_preprocess(df)
        self.data_info = get_data_info(self.data,cat_vars)
                         

    def lawsch_preprocess(self,dataset):
        dataset.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
        dataset.dropna(axis=0, inplace=True, subset=['admit'])
        dataset.replace(to_replace='', value=np.nan, inplace=True)
        dataset.dropna(axis=0, inplace=True)
        dataset = dataset[dataset['race'] != 'Asian']

        for col in dataset.columns:
            if dataset[col].isnull().sum() > 0:
                dataset.drop(col, axis=1, inplace=True)

        con_vars = ['lsat','gpa']
        cat_vars = [col for col in dataset.columns if col not in con_vars]
        return dataset,cat_vars




def load_lawschool_for_gan(batch_size=500,train_num=15400,valid_num=10000,test_num=10000):
    lawschool = LawschoolDataset()
    datasets= lawschool.data
    data_info = lawschool.data_info
    tf = BaseTransformer(datasets, data_info)
    data = tf.transform()
    feature = torch.tensor(data.astype(np.float32))
    tensordataset = TensorDataset(feature)
    set1, set2, set3,_ = torch.utils.data.random_split(
                tensordataset,
                lengths=[
                    train_num,
                    valid_num,
                    test_num,
                    len(tensordataset)-train_num-valid_num-test_num,
                ],
                generator=torch.Generator().manual_seed(42)
            )
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader,data_info,tf

def load_lawschool_for_clf(batch_size=500,train_num=15400,valid_num=10000,test_num=10000):

    lawschool = LawschoolDataset()
    datasets= lawschool.data
    data_info = lawschool.data_info
    tf = BaseTransformer(datasets, data_info)
    data = tf.transform()
    X= data[:,:-2]
    y = np.argmax(data[:,-2:],axis=1)
    targets = torch.tensor(y, dtype=torch.long)
    feature = torch.tensor(X.astype(np.float32))

    tensordataset = TensorDataset(feature, targets)
    set1, set2, set3,_ = torch.utils.data.random_split(
                tensordataset,
                lengths=[
                    train_num,
                    valid_num,
                    test_num,
                    len(tensordataset)-train_num-valid_num-test_num,
                ],
                generator=torch.Generator().manual_seed(42)
            )
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader