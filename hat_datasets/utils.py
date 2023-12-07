import numpy as np
import torch
import pandas as pd
import random
from torch.distributions import Categorical,Normal

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic =True
    random.seed(seed)

def save_generated_data(data,name,column_name):
    data_pd = pd.DataFrame(data,index=None,columns = column_name)
    data_pd.to_csv('./GenerateData/'+ name+'.csv', index=False)


def data_generator():

    gaussian = torch.randn((5000,1))
    gaussian = (gaussian-gaussian.min())/(gaussian.max()-gaussian.min())


    m2 = Categorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
    cat  = m2.sample(sample_shape=torch.Size([5000,]))
    onehot = torch.zeros(5000,4)
    onehot[torch.arange(5000),cat] =1

    return torch.cat((onehot,gaussian),dim=1)


#data：pandas dataframe
def get_data_info(data,categorical_columns):
    data_info = []
    con_columns = [col for col in data.columns if col not in categorical_columns]
    for column in con_columns:
        data_info.append((1,'tanh', column))
    for column in categorical_columns:
        a = (data[column].unique().shape[0],'softmax',column)
        data_info.append(a)

    return data_info  


def get_column_loc(data_info):
    tanh_list = []
    soft_list =[]
    st=0
    for item in data_info:
        if item[1]=='tanh':
            tanh_list.append(st)
            st = st + item[0]
        elif item[1]=='softmax':
            soft_list.append(st)
            st = st + item[0]
        else:
            assert 0
    return tanh_list,soft_list

#return numpy array
def convert_onehot(data):
    onehot_data = pd.get_dummies(data,columns = data.columns ).values   
    return onehot_data             

def lawsch_postprocess(data):
    data['lsat'] = data['lsat'].astype('float').round(decimals=0)
    data['gpa'] = data['gpa'].astype('float')
    data['gpa'] = data['gpa'].round(decimals=2)
    return data

def compas_postprocess(data):
    data.loc[(data["diff_jail"] < 0 ,"diff_jail")]= 0
    data[['age','diff_custody','diff_jail','priors_count']] = data[['age','diff_custody','diff_jail','priors_count']].astype('float')
    data[['age','diff_custody','diff_jail','priors_count']] = data[['age','diff_custody','diff_jail','priors_count']].round(decimals=0)

    return data

def adult_postprocess(data):
    data['Age'] = data['Age'].astype('float').round(decimals=0)
    data['EducationNum'] = data['EducationNum'].astype('float64')
    data['HoursPerWeek'] = data['HoursPerWeek'].astype('float64')
    return data


def health_postprocess(data):
    col =['LabCount_total','LabCount_months','DrugCount_total','DrugCount_months','PayDelay_total','PayDelay_max','PayDelay_min']
    data[col]=data[col].astype('float')
    data = data.round({'LabCount_total':0,'LabCount_months':0, 'DrugCount_total':0,
           'DrugCount_months':0, 'PayDelay_total':0, 'PayDelay_max':0, 'PayDelay_min':0})
    col=['AgeAtFirstClaim','Sex','max_CharlsonIndex']
    data[col]=data[col].astype('int')
    
    return data