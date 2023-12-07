from src.wt_train_mean import multitask_Trainer,eval_task1,eval_task2
import numpy as np

#generate partition key
arr = np.arange(10)
np.random.seed(0)
partition = np.random.choice(arr, size=5, replace=False)

# set maximal wt_bit 
wt_bit = 3

DATASET = 'cifar10'
Net_arch = 'resnet'
delta = 0.01 # parameters that tradeoff between main loss and wt loss

# choose a wt_task, default: "Rotate90"
wt_task = 'Rotate90'

#support load self-defined watermark_loader
watermark_loader =None

seed = 0

trainer_base =  multitask_Trainer(DATASET,Net_arch,watermark_loader,wt_task,wt_bit,delta,partition,seed,device_ids=[1])
trainer_base.run(load_best=False, retrain=True) # retrain=True means train a model from scratch
test_acc = eval_task1(trainer_base.models['C'],trainer_base.test_loader,trainer_base.device,Net_arch)  
wt_acc,_,_ = eval_task2(trainer_base.models['C'],trainer_base.wt_loader,trainer_base.nz,partition,wt_bit,trainer_base.device,Net_arch)
print(test_acc,wt_acc)
