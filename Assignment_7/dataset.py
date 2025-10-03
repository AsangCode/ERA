import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label

def get_initial_dataloaders(batch_size=128):
    """Get dataloaders with basic transforms for mean/std calculation"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    basic_transform = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    return get_dataloaders(
        batch_size=batch_size,
        transform_train=basic_transform,
        transform_test=basic_transform
    )

def get_dataloaders(batch_size, transform_train, transform_test):
    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True
    )
    testset = datasets.CIFAR10(
        root='./data', train=False, download=True
    )

    train_dataset = CIFAR10Dataset(trainset, transform_train)
    test_dataset = CIFAR10Dataset(testset, transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
