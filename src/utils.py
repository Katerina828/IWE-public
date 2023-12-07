import torch
import numpy as np
import random
from torch.optim.lr_scheduler import _LRScheduler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic =True
    random.seed(seed)

def compute_moving_average(array,alpha):
    smoothed_curve = torch.zeros_like(torch.tensor(array))

    # 计算指数移动平均
    for t in range(len(array)):
        if t == 0:
            smoothed_curve[0] = array[0]
        else:
            smoothed_curve[t] = alpha * smoothed_curve[t-1] + (1 - alpha) * array[t]
    return smoothed_curve

def compute_acc(trainer, train_loader,test_loader):
    net = trainer.models['C']
    net.eval()
    correct = 0.0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(trainer.device), y.to(trainer.device)
            N = len(x)
            scores = net(x)
            pred_y = torch.argmax(scores, dim=1)
            correct += torch.sum(pred_y == y).item()
    N = len(train_loader.dataset)
    train_acc = correct / N
    print('train acc: {}'.format(train_acc))

    correct = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(trainer.device), y.to(trainer.device)
            N = len(x)
            scores = net(x)
            pred_y = torch.argmax(scores, dim=1)
            correct += torch.sum(pred_y == y).item()
    N = len(test_loader.dataset)
    test_acc = correct / N
    print('test acc: {}'.format(test_acc))
    diff = train_acc-test_acc
    print('acc diff: {}'.format(round(diff,4)))
    return train_acc, test_acc
    

def get_data_info(data,categorical_columns):
    data_info = []
    con_columns = [col for col in data.columns if col not in categorical_columns]
    for column in con_columns:
        data_info.append((1,'tanh', column))
    for column in categorical_columns:
        a = (data[column].unique().shape[0],'softmax',column)
        data_info.append(a)

    return data_info  

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]