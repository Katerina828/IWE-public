from scipy.stats import combine_pvalues, ttest_ind_from_stats, ttest_ind
import scipy.stats as stats
import torch
import numpy as np


def get_p(outputs_train, outputs_test,alternative):
    if type(outputs_train)!=type(np.zeros(1)):
        pred_test = outputs_test.detach().cpu().numpy()
        pred_train = outputs_train.detach().cpu().numpy()
    else:
        pred_test = outputs_test
        pred_train = outputs_train
    tval, pval = ttest_ind(pred_test, pred_train, alternative=alternative, equal_var=False)
    if pval < 0:
        raise Exception(f"p-value={pval}")
    return pval

def get_p_values(num_ex, train, test, k,alternative):
    total = train.shape[0]
    sum_p = 0  
    p_values = []
    positions_list = []
    for i in range(k):
        positions = torch.randperm(total)[:num_ex]
        p_val = get_p(train[positions], test[positions],alternative)
        positions_list.append(positions)
        p_values.append(p_val)
    return p_values
