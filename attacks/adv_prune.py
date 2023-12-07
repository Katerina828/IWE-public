import sys
sys.path.append('/workspace/mytoolbox')
import marveltoolbox as mt 
import os
import time
import torch
from models.resnet_noisy import resnet18
from models.anp_batchnorm import NoisyBatchNorm1d,NoisyBatchNorm2d
import numpy as np
from collections import OrderedDict
from src.wt_train_advprune import eval_task1,eval_task2
import pandas as pd

anp_eps = 0.4
anp_steps = 1
output_dir = 'result/cifar100/mask/'
learning_rate = 0.2
print_every = 500
nb_iter = 2000
anp_alpha = 0.2
pruning_max=0.9
pruning_step=0.05
Net_arch = 'resnet'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_adv_mask(model_checkpoint,data_loader,test_loader,wt_loader,partition,wt_bit):
    checkpoint = torch.load(model_checkpoint, map_location=device)
    net= resnet18(num_classes=100, norm_layer=NoisyBatchNorm2d).to(device)
    load_state_dict(net, orig_state_dict=checkpoint)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    acc1 = eval_task1(net,test_loader,device,Net_arch) 
    print(f"baseline acc:{acc1}")

    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=learning_rate, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=anp_eps / anp_steps)

    # Step 3: train backdoored modelsprint('Iter \t lr \t Time \t MainTaskAcc \t WtTaskAcc')
    nb_repeat = int(np.ceil(nb_iter / print_every))
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=data_loader,
                                           mask_opt=mask_optimizer, noise_opt=noise_optimizer)
        acc1 = eval_task1(net,test_loader,device,Net_arch)  
        acc2,_,_ = eval_task2(net,wt_loader,10,partition,wt_bit,device,Net_arch)
        end = time.time()
       
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f}'.format((i + 1) * print_every, lr, end - start, acc1, acc2))

    save_mask_scores(net.state_dict(), os.path.join(output_dir, 'mask_values.txt'))


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)

def prune_by_number(model,mask_values,pruning_max,pruning_step,test_loader,wt_loader,partition,wt_bit):
    results =[]
    nb_max = int(np.ceil(pruning_max*len(mask_values)))
    nb_step = int(np.ceil(pruning_step*len(mask_values)))
    for start in range(0,nb_max,nb_step):
        for i in range(start,start+nb_step):
            pruning(model,mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        acc1 = eval_task1(model,test_loader,device,Net_arch)  
        acc2,_,_ = eval_task2(model,wt_loader,10,partition,wt_bit,device,Net_arch)
        print('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f}'.format(
            start+1, layer_name, neuron_idx, value, acc1, acc2))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f}'.format(
            start+1, layer_name, neuron_idx, value, acc1, acc2))
    return results


def load_state_dict(net, orig_state_dict):
    if 'model_C' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['model_C']


    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=anp_eps)


def mask_train(model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = anp_alpha * loss_nat + (1 - anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc





def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


def adv_prune(chkpt_file,stealer_loader,test_loader,wt_loader,pruning_max,pruning_step,output_dir,partition,wt_bit):
    compute_adv_mask(chkpt_file,stealer_loader,test_loader,wt_loader,partition,wt_bit)
    checkpoint = torch.load(chkpt_file, map_location=device)
    net = resnet18(num_classes=100, norm_layer=None).to(device)
    load_state_dict(net, orig_state_dict=checkpoint)
    mask_values = read_data(os.path.join(output_dir, 'mask_values.txt'))
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    print('No. \t Layer Name \t Neuron Idx \t Mask \t Acc1 \t Acc2 ')
    results = prune_by_number(net,mask_values,pruning_max,pruning_step,test_loader,wt_loader,partition,wt_bit)
    result_df = pd.DataFrame([x.split('\t') for x in results])
    result_df['per'] = np.arange(0,pruning_max,pruning_step)
    result_df = result_df.rename(columns = {4:'Task1_ACC',5:'Task2_ACC'})
    return result_df
