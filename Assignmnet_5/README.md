# MNIST Classification with PyTorch

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification that achieves >99.4% test accuracy while keeping the parameter count under 20k.

## Model Architecture Analysis

### Parameter Count
Total trainable parameters: 14,768

### Key Components Used
1. **Batch Normalization**
   - Used after every convolutional layer
   - Helps with training stability and faster convergence
   - Implementation can be found in layers:
     ```python
     self.bn1 = nn.BatchNorm2d(10)
     self.bn2a = nn.BatchNorm2d(10)
     self.bn2b = nn.BatchNorm2d(20)
     self.bn2c = nn.BatchNorm2d(16)
     self.bn3 = nn.BatchNorm2d(24)
     self.bn4 = nn.BatchNorm2d(32)
     self.bn5 = nn.BatchNorm2d(24)
     ```

2. **Dropout**
   - Strategic placement at multiple levels for effective regularization
   - Dropout rate: 0.12
   - Implementation:
     ```python
     self.dropout = nn.Dropout(dropout_p)
     ```
   - Applied at three strategic points in the network:
     - After initial feature extraction
     - After mid-level features
     - Before final classification

3. **Global Average Pooling (GAP)**
   - Used instead of multiple fully connected layers
   - Reduces parameters while maintaining performance
   - Implementation:
     ```python
     self.gap = nn.AdaptiveAvgPool2d(1)
     ```

## Model Performance

### Test Accuracy
- Final Test Accuracy: **99.43%**
- Best Validation Accuracy: 99.26%
- Training completed in 20 epochs

### Training Parameters
- Batch Size: 32
- Learning Rate: 0.002
- Weight Decay: 0.00015
- Dropout Rate: 0.12

## Architecture Details

The model uses a carefully designed CNN architecture with:
1. Progressive channel expansion (10 → 20 → 16 → 24 → 32 → 24)
2. Skip connection in early layers
3. 1x1 convolution for efficient channel reduction
4. Two-stage pooling for spatial reduction
5. Batch normalization after each convolution
6. Strategically placed dropout layers

## Training Curve
```
Epoch 01/20  train_loss=0.3347 train_acc=91.792%  val_loss=0.1370 val_acc=96.260%
Epoch 05/20  train_loss=0.0458 train_acc=98.670%  val_loss=0.0529 val_acc=98.500%
Epoch 10/20  train_loss=0.0299 train_acc=99.098%  val_loss=0.0398 val_acc=98.870%
Epoch 15/20  train_loss=0.0176 train_acc=99.498%  val_loss=0.0299 val_acc=99.140%
Epoch 20/20  train_loss=0.0116 train_acc=99.716%  val_loss=0.0252 val_acc=99.260%
```

## Requirements Met
- [x] Under 20k parameters (14,768)
- [x] Test accuracy > 99.4% (99.43%)
- [x] Uses Batch Normalization
- [x] Uses Dropout
- [x] Uses Global Average Pooling
- [x] Completes training in 20 epochs
