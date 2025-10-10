# ResNet18 Training on CIFAR-100

This project implements and trains a ResNet18 model from scratch on the CIFAR-100 dataset, targeting 73% top-1 accuracy.

## Model Architecture

The implementation uses ResNet18 architecture with the following modifications for CIFAR-100:
- Initial convolution layer uses 3×3 kernels with stride 1 (instead of 7×7 with stride 2)
- No initial max pooling layer
- Final average pooling adjusted for CIFAR-100's 32×32 input size
- Output layer modified for 100 classes

### Network Structure
```
ResNet18
├── Initial Conv Layer (3×3, 64 channels)
├── Layer1: 2×BasicBlock (64 channels)
├── Layer2: 2×BasicBlock (128 channels)
├── Layer3: 2×BasicBlock (256 channels)
├── Layer4: 2×BasicBlock (512 channels)
└── Final Linear Layer (512 → 100 classes)
```

## Training Setup

### Requirements
```
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
numpy>=1.24.0
```

### Installation
```bash
pip install -r requirements.txt
```

### Training the Model
```bash
python train.py --lr 0.1 --batch-size 128 --epochs 100
```

### Training Parameters
- **Optimizer**: SGD with momentum (0.9) and weight decay (5e-4)
- **Learning Rate**: OneCycleLR scheduler with max_lr=0.1
- **Batch Size**: 128 (default)
- **Epochs**: 100 (default)

### Data Augmentation
The training pipeline includes the following augmentations:
- Random Crop (32×32 with padding=4)
- Random Horizontal Flip
- Random Rotation (±15 degrees)
- Color Jitter (brightness=0.2, contrast=0.2, saturation=0.2)
- Normalization (using CIFAR-100 mean and std values)

## Project Structure
```
.
├── model.py          # ResNet18 model implementation
├── train.py          # Training script with data loading and training loop
└── requirements.txt  # Project dependencies
```

## Features
- Automatic checkpoint saving for best accuracy
- Training progress visualization with tqdm
- GPU support with DataParallel
- Resume training capability from checkpoints
- Proper validation loop with accuracy tracking

## Training Process
The model is trained with:
1. OneCycleLR learning rate scheduling for faster convergence
2. Strong data augmentation to prevent overfitting
3. Proper model checkpointing to save the best model
4. Progress tracking with both training and validation metrics
5. Efficient data loading with multiple workers

## Checkpoints
Checkpoints are automatically saved in the `checkpoint` directory when the model achieves a new best accuracy. To resume training from a checkpoint:
```bash
python train.py --resume
```

## Results
The model is designed to achieve 73% top-1 accuracy on CIFAR-100 test set within 100 epochs of training.

## Hardware Requirements
- CUDA-capable GPU recommended for faster training
- At least 8GB RAM
- Training time varies based on hardware (approximately 2-3 hours on a modern GPU)
