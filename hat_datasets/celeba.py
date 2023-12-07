
import torch
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
root_dir='C:/Users/Liuyifan/DATASET/CelebA'

    
def load_celeba(batch_size: int = 64, img_size: int = 64):
    """
    Load CelebA dataset (download if necessary) and split data into training,
        validation, and test sets.
    Args:
        downsample_pct: the proportion of the dataset to use for training,
            validation, and test
        train_pct: the proportion of the downsampled data to use for training
    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """

    transform = transforms.Compose([
                               transforms.Resize((224,224)),
                               transforms.CenterCrop(178),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = CelebA(root=root_dir, split='train',target_type = 'attr', download=False, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=4)

    valset = CelebA(root=root_dir, split='valid',target_type = 'attr', download=False, transform=transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True,num_workers=4)
    
    testset = CelebA(root=root_dir, split='test',target_type = 'attr', download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)


    return train_loader, val_loader, test_loader, test_loader
