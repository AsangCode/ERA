import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import CIFAR10Net
from dataset import get_dataloaders
from transforms import Transforms
from utils import AverageMeter, plot_metrics, get_mean_std

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        
        train_loss.update(loss.item(), data.size(0))
        train_acc.update(100. * correct / data.size(0), data.size(0))
        
        pbar.set_postfix({'Loss': f'{train_loss.avg:.4f}', 
                         'Acc': f'{train_acc.avg:.2f}%'})
    
    return train_loss.avg, train_acc.avg

def test_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            test_loss.update(loss.item(), data.size(0))
            test_acc.update(100. * correct / data.size(0), data.size(0))
    
    print(f'\nTest set: Average loss: {test_loss.avg:.4f}, '
          f'Accuracy: {test_acc.avg:.2f}%\n')
    
    return test_loss.avg, test_acc.avg

def main(model=None, train_loader=None, test_loader=None, device=None, 
         num_epochs=50, learning_rate=0.001):
    """
    Main training function that can be called from notebook or command line
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If model and data loaders are not provided, create them
    if model is None or train_loader is None or test_loader is None:
        # Get dataloaders
        temp_loader, _ = get_initial_dataloaders(batch_size=128)
        mean, std = get_mean_std(temp_loader)
        
        transforms = Transforms(mean.tolist(), std.tolist())
        train_loader, test_loader = get_dataloaders(
            batch_size=128,
            transform_train=transforms.train_transforms,
            transform_test=transforms.test_transforms
        )
        
        # Initialize model
        model = CIFAR10Net().to(device)
    
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = test_epoch(
            model, test_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Plot metrics
        if (epoch + 1) % 5 == 0:
            plot_metrics(train_losses, train_accs, test_losses, test_accs)

if __name__ == '__main__':
    main()
