from .adult import load_adult_for_gan, load_adult_for_clf
from .lawschool import load_lawschool_for_clf, load_lawschool_for_gan
from .compas import load_compas_for_clf, load_compas_for_gan
from .health import load_health_for_clf, load_health_for_gan
from .purchase import load_purchase
from .fashion_mnist import load_fmnist, load_fmnist_pairs
from .mnist import load_mnist, load_mnist_pairs
from .cifar import load_cifar10, load_cifar100
from .svhn import load_svhn
from .celeba import load_celeba
from .caltech101 import load_caltech101

def load_data(dataset,model): 
    if model=='clf':
        if dataset == 'adult':
            return load_adult_for_clf()
        elif dataset =='lawschool':
            return load_lawschool_for_clf()
        elif dataset =='compas':
            return load_compas_for_clf()
        elif dataset =='health':
            return load_health_for_clf()
        elif dataset =='purchase':
            return load_purchase()
            
    elif model =='gan':
        if dataset == 'adult':
            return load_adult_for_gan()
        elif dataset =='lawschool':
            return load_lawschool_for_gan()
        elif dataset =='compas':
            return load_compas_for_gan()
        elif dataset =='health':
            return load_health_for_gan()
    else:
        return [None, None, None, None]
    
def load_image(dataset, all_frac, train_frac,steal_frac, batch_size, img_size, label_list, is_norm=False): 
    if dataset == 'mnist':
        return load_mnist(all_frac, train_frac, batch_size, img_size=img_size,  label_list=label_list, is_norm=is_norm)
    elif dataset == 'svhn':
        return load_svhn(all_frac, train_frac, batch_size, img_size=img_size,  label_list=label_list, is_norm=is_norm)
    elif dataset == 'fmnist':
        return load_fmnist(all_frac, train_frac, batch_size, img_size=img_size,  label_list=label_list, is_norm=is_norm)
    elif dataset == 'cifar10':
        return load_cifar10(all_frac, train_frac,steal_frac, batch_size,img_size=img_size, label_list=label_list, is_norm=is_norm)
    elif dataset == 'cifar100':
        return load_cifar100(all_frac,train_frac,steal_frac,batch_size,label_list=label_list,is_norm=is_norm)
    elif dataset == 'celeba':
        return load_celeba(batch_size=batch_size,img_size=img_size)
    elif dataset =='caltech101':
        return load_caltech101(train_pct =train_frac,steal_pct= steal_frac,batch_size=batch_size,img_size=img_size,label_list=label_list)

    else:
        return [None, None, None, None]