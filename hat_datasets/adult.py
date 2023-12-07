import torch
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import random
from random import shuffle
import pandas as pd
import numpy as np
import pandas as pd
from os import path


from hat_datasets.utils import get_data_info
from hat_datasets.transformer import BaseTransformer
import torch
import operator
from sklearn.model_selection import train_test_split


class AdultDataset():
    
    def __init__(self):
        self.data = pd.read_csv('./hat_datasets/rawdata/combined_set.csv')
        self.preprocess()
        self.data_info = get_data_info(self.data, self.cat_vars)
        
        #self.train_valid_data, self.test_data = train_test_split(self.data, test_size=0.3, random_state=3)

        
    def preprocess(self):
        CapitalGainLoss(self.data)
        NativeCountry(self.data)
        MaritalStatus(self.data)
        #discretizer = KBinsDiscretizer(n_bins=45, encode='ordinal', strategy='uniform')
        #data['HoursPerWeek'] = discretizer.fit_transform(data['HoursPerWeek'].values.reshape(-1,1))
        #data['HoursPerWeek'] = pd.cut(data['HoursPerWeek'],bins=45,labels=False)
        self.label = 'Income'
        self.con_vars = ['Age']
        self.cat_vars = [col for col in self.data.columns if col not in self.con_vars]
        self.columns_name = self.con_vars + self.cat_vars
        #cat_vars.remove(label)
        #X, y = preprocess_data(data,cat_vars,label)  

              

def load_adult_for_gan(batch_size=500,train_num=25000,valid_num=10000,test_num=10000):
    adult = AdultDataset()
    datasets= adult.data
    data_info = adult.data_info
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
                generator=torch.Generator().manual_seed(1)
            )
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader,data_info,tf

def load_adult_for_clf(batch_size=500,train_num=30000,valid_num=10000,test_num=9000):

    adult = AdultDataset()
    datasets= adult.data
    data_info = adult.data_info
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
                generator=torch.Generator().manual_seed(1)
            )
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=False, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader



def CapitalGainLoss(data):

    data.loc[(data["CapitalGain"] > 7647.23),"CapitalGain"] = 'high'
    data.loc[(data["CapitalGain"] == 0 ,"CapitalGain")]= 'zero'
    data.loc[operator.and_(data["CapitalGain"]!='zero', data["CapitalGain"]!='high' ),"CapitalGain"] = 'low'

    data.loc[(data["CapitalLoss"] > 1874.19),"CapitalLoss"] = 'high'
    data.loc[(data["CapitalLoss"] == 0 ,"CapitalLoss")]= 'zero'
    data.loc[operator.and_(data["CapitalLoss"]!='zero', data["CapitalLoss"]!='high'),"CapitalLoss"] = 'low'


    #NativeCountry 41---> 2
def NativeCountry(data):
    
    datai = [data]

    for dataset in datai:
        dataset.loc[dataset["NativeCountry"] != ' United-States', "NativeCountry"] = 'Non-US'
        dataset.loc[dataset["NativeCountry"] == ' United-States', "NativeCountry"] = 'US'


# MaritalStatus  7 --->2
def MaritalStatus(data):
    
    data["MaritalStatus"] = data["MaritalStatus"].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
    data["MaritalStatus"] = data["MaritalStatus"].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

# Age 74 dimention   
# HoursPerWeek 96 dimention
def Discretization(data):
    data['Age']= pd.cut(data['Age'],bins=35)
    data['HoursPerWeek'] = pd.cut(data['HoursPerWeek'],bins=45)




def ordinal_encode(enc,data):
    colomns = [col for col in data.columns if data[col].dtype=="object"]
    data[colomns] = enc.fit_transform(data[colomns])
    return data