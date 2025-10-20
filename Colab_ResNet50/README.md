# ResNet50 Training on TinyImageNet

This project implements training of ResNet50 from scratch on the TinyImageNet dataset, aiming to achieve 75% top-1 accuracy.

## Dataset
TinyImageNet is a subset of ImageNet with:
- 200 classes
- 500 training images per class
- 50 validation images per class
- Images are 64x64 pixels

## Project Structure
- `dataset.py`: Dataset loading and preprocessing
- `model.py`: ResNet50 model implementation
- `train.py`: Training loop and utilities
- `requirements.txt`: Project dependencies

## Setup and Training

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

The script will:
- Download and extract TinyImageNet dataset
- Train ResNet50 from scratch
- Save best model checkpoints
- Log metrics to TensorBoard

## Training Details
- Batch size: 128
- Optimizer: SGD with momentum (0.9)
- Initial learning rate: 0.1
- Weight decay: 1e-4
- Learning rate schedule: Cosine Annealing
- Number of epochs: 100

## Data Augmentation
- Random resized crop
- Random horizontal flip
- Color jitter
- Normalization using ImageNet statistics

## Monitoring
Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir runs
```

## Model Checkpoints
Best model checkpoints are saved in the `checkpoints` directory.
