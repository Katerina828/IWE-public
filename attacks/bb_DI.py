import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import hmean

def ns_pvalues(df, mode='loss',per=1.0,start=5,stop=100,step=10):
    if mode=='loss' or mode=='entropy'or mode =='Tmargin':
        alternative='less'
        ascending=True
    elif mode == 'max_posterior' or mode =='margin':
        alternative='greater'
        ascending=False
    result=dict()
    mem = df[df['label']==1]
    nonmem = df[df['label']==0]
    mem = mem[mode].copy()
    nonmem = nonmem[mode].copy()
    N = int(per*len(mem))
    mem.sort_values(ascending=ascending, inplace=True)
    mem = mem.reset_index(drop=True)[:N]
    #diff = mem.mean()-nonmem.mean()
    for ns in range(start,stop,step):
        hm_list =[]
        for i in range(10):
            p_list = compute_Pvalues(ns,mem,nonmem,inner_loop=100,alternative=alternative)
            hm_p = hmean(p_list) 
            hm_list.append(hm_p)
        result[ns]=hm_list
    df = pd.DataFrame.from_dict(result).stack().reset_index()
    df = df.drop(columns =['level_0'])
    df = df.rename(columns={"level_1": "ns", 0: "Pvalues"})
    return df


def compute_Pvalues(ns,mem,nonmem,inner_loop,alternative):
    #per: top "per" percentile confidence training data
    p_list =[]
    for i in range(inner_loop):
        sample_mem = sample_data(mem.values,ns)
        sample_nonmem = sample_data(nonmem.values,ns)
        _, pval = ttest_ind(sample_mem,sample_nonmem, alternative=alternative, equal_var=False)
        if pval < 0:
            raise Exception(f"p-value={pval}")
        if pval>0:
            p_list.append(pval)
    return p_list

def sample_data(data,ns):
    idx = np.random.permutation(len(data))[:ns]
    sample_data = data[idx]
    return sample_data
