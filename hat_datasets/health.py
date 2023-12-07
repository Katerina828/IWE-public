import numpy as np
import pandas as pd
from os import path
from urllib import request
from sklearn.preprocessing import MinMaxScaler
from hat_datasets.utils import get_data_info
import zipfile
import matplotlib.pyplot as plt
import math
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from hat_datasets.transformer import BaseTransformer

class HealthDataset():
    
    def __init__(self):

        print("load Health...")
        df_health = pd.read_csv('./hat_datasets/rawdata/health_without_year.csv')
        
        #basic preprocess
        df_health = process_health_per_year(df_health)
        #discretization, 
        discretization(df_health)

        self.con_vars = ['LabCount_total','LabCount_months','DrugCount_total','DrugCount_months'
          ,'PayDelay_total','PayDelay_max','PayDelay_min']
        
        #self.cat_vars = [col for col in df_health.columns if col not in self.con_vars]
        self.cat_vars = [col for col in df_health.columns if '=' in col]
        self.cat_vars.extend(['AgeAtFirstClaim','Sex','max_CharlsonIndex'])
        

        self.columns_name = self.con_vars + self.cat_vars
        self.data  = df_health[self.columns_name]
        
        #get data info
        self.data_info = get_data_info(self.data, self.cat_vars)
        self.label = 'max_CharlsonIndex'


def load_health_for_gan(batch_size: int = 500,train_num=100000,valid_num=3000,test_num=10000):
    health = HealthDataset()
    datasets= health.data
    data_info = health.data_info
    tf = BaseTransformer(datasets, data_info)
    data = tf.transform()
    feature = torch.tensor(data.astype(np.float32))
    adultTensorDataset = TensorDataset(feature)
    set1, set2, set3,_ = torch.utils.data.random_split(
                adultTensorDataset,
                lengths=[
                    train_num,
                    valid_num,
                    test_num,
                    len(adultTensorDataset)-train_num-valid_num-test_num,
                ],
                generator=torch.Generator().manual_seed(1)
            )
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=False, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader,data_info,tf

def load_health_for_clf(batch_size: int = 500,train_num=40000,valid_num=60000,test_num=10000):

    health = HealthDataset()
    datasets= health.data
    data_info = health.data_info
    tf = BaseTransformer(datasets, data_info)
    data = tf.transform()
    X= data[:,:-2]
    y = np.argmax(data[:,-2:],axis=1)
    targets = torch.tensor(y, dtype=torch.long)
    feature = torch.tensor(X.astype(np.float32))

    tensordataset = TensorDataset(feature, targets)
    print(len(tensordataset))
    set1, set2, set3,_ = torch.utils.data.random_split(
                tensordataset,
                lengths=[
                    train_num,
                    valid_num,
                    test_num,
                    len(tensordataset)-train_num-valid_num-test_num,
                ],
                generator=torch.Generator().manual_seed(1)
            )
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=False, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader




def process_health_per_year(health_year):

    health_year['max_CharlsonIndex'] = health_year['max_CharlsonIndex'].replace([2,4,6],1)
    health_year['max_CharlsonIndex'] = 1 - health_year['max_CharlsonIndex']
    map_x = {'0-9': 0, '10-19': 1, '20-29':2, '30-39':3,
         '40-49':4, '50-59':5, '60-69':6,'70-79':7, '80+':8, '?':9}
    health_year['AgeAtFirstClaim'] = health_year['AgeAtFirstClaim'].map(map_x)
    map_y = {'?':0,'F':1,'M':2}
    health_year['Sex'] = health_year['Sex'].map(map_y)
    health_year.drop([ 'MemberID'], axis=1, inplace=True)
    #health_year.drop(['Year', 'MemberID'], axis=1, inplace=True)
    return health_year


def discretization(df_health):
    cat_vars = [col for col in df_health.columns if '=' in col]
    for column in cat_vars:
        max_item = df_health[column].max()+1
        min_item = df_health[column].min()-1
        q1 = df_health[column].quantile(.6) + 0.5
        q2 = df_health[column].quantile(.85) + 0.5
        if q2 ==q1:
            q2 +=1
        bins = pd.IntervalIndex.from_tuples([(min_item, q1,), (q1, q2), (q2, max_item)])
        df_health[column] = pd.cut(df_health[column], bins)

def ordinal_encode(enc,data):
    colomns = [col for col in data.columns if data[col].dtype=="object"]
    data[colomns] = enc.fit_transform(data[colomns])
    return data