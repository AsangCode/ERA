import os
import math
import torch
from torch.amp import autocast, GradScaler
from lr_finder import LRFinder
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ResNet50
from dataset import get_data_loaders

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    writer,
    checkpoint_dir
):
    # Create gradient scaler for AMP
    scaler = GradScaler()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass and optimize with AMP
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate (OneCycleLR is updated every step)
            scheduler.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'acc': f"{(running_corrects.double() / total_samples).item()*100:.2f}%"
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%')
        
        # Log training metrics
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass with AMP
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%')
        
        # Log validation metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save checkpoint if best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
    
    return model

def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(batch_size=128)  # Smaller batch size for better generalization
    
    # Create model with adjusted BN momentum
    model = ResNet50().to(device)
    
    # Use torch.compile() for better performance
    if torch.cuda.is_available():
        model = torch.compile(model)
    
    # Adjust batch norm momentum
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1  # Lower momentum for more stable statistics
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)  # Higher initial LR for finder
    
    # Run LR finder
    print("Running learning rate finder...")
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=200, smooth_f=0.05)
    
    # Get the learning rate with minimum loss
    min_loss_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(min(lr_finder.history['loss']))]
    suggested_lr = min_loss_lr / 10  # Usually divide by 10 for stable training
    
    print(f"\nLR Finder Results:")
    print(f"Minimum loss achieved at learning rate: {min_loss_lr:.6f}")
    print(f"Suggested learning rate (min_loss_lr/10): {suggested_lr:.6f}")
    
    # Plot the results
    lr_finder.plot()
    
    # Get the model and optimizer back to their initial states
    lr_finder.reset()
    
    # Use the suggested learning rate
    init_lr = suggested_lr
    
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)
    
    # Use OneCycleLR policy with more aggressive initial phase
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=init_lr * 1.5,  # Slightly higher max_lr for better exploration
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Faster warmup
        div_factor=10,  # Less conservative initial LR
        final_div_factor=1e3  # Less aggressive final LR reduction
    )
    
    # TensorBoard writer
    writer = SummaryWriter('runs/resnet50_training')
    
    # Train the model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        writer=writer,
        checkpoint_dir='checkpoints'
    )
    
    writer.close()

if __name__ == '__main__':
    main()