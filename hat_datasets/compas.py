import numpy as np
import pandas as pd
from os import path
from hat_datasets.utils import get_data_info
#from utils import save_generated_data
import torch
from hat_datasets.transformer import BaseTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
#-----------------------------
#Compas: orginal :(7214, 53)
# After preproces: (5278, 11)
# predict two_year_recid, 
# Task: binary classification

def compas_preprocess(df):

    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != '0']
    df = df[df['score_text'] != 'N/A']

    df['in_custody'] = pd.to_datetime(df['in_custody'])
    df['out_custody'] = pd.to_datetime(df['out_custody'])
    df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.days
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.days

    df.drop(
        [
            'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
            'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
        ], axis=1, inplace=True
    )
    df = df[df['race'].isin(['African-American', 'Caucasian'])]

    features = df.drop(['is_recid', 'is_violent_recid', 'violent_recid', 'two_year_recid'], axis=1)
    labels = 1 - df['two_year_recid']

    features = features[[
        'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree',
        'v_score_text'
    ]]

    data = pd.concat([features,labels],axis = 1)
    data[['juv_fel_count','two_year_recid']] = data[['juv_fel_count','two_year_recid']].astype('object')
    con_vars = [i for i in data.columns if data[i].dtype=='int64'or data[i].dtype=='float64']
    cat_vars = [col for col in data.columns if col not in con_vars]
    return data, cat_vars

def ordinal_encode(enc,data):
    colomns = [col for col in data.columns if data[col].dtype=="object"]
    data[colomns] = enc.fit_transform(data[colomns])
    return data

class CompasDataset():
    def __init__(self):
        data_dir = 'hat_datasets/rawdata'
        data_file = path.join(data_dir,'compas-scores-two-years.csv')
        df = pd.read_csv(data_file)
        self.data, self.cat_vars = compas_preprocess(df)
        self.data_info = get_data_info(self.data ,self.cat_vars)
        #self.con_vars = ['age','diff_custody','diff_jail','priors_count']
        #self.label = 'two_year_recid'
    

def load_compas_for_gan(batch_size: int = 100,train_num: int = 2700,valid_num: int = 1100,test_num: int = 1400):
    compas = CompasDataset()
    datasets= compas.data
    data_info = compas.data_info
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
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader,data_info,tf

def load_compas_for_clf(batch_size: int = 100,train_num: int = 2700,valid_num: int = 1100,test_num: int = 1400):

    compas = CompasDataset()
    datasets= compas.data
    data_info = compas.data_info
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
    train_loader = DataLoader(set1, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    valid_loader = DataLoader(set2, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(set3, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
    return train_loader,valid_loader,test_loader
    
