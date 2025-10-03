# CIFAR-10 Custom CNN Implementation

This project implements a custom CNN architecture for the CIFAR-10 dataset with specific architectural requirements and advanced data augmentation techniques.

## Architecture Highlights

1. **No MaxPooling**: Instead uses strided convolutions and dilated convolutions for downsampling
2. **Depthwise Separable Convolution**: Implemented in one of the layers for efficient computation
3. **Dilated Convolution**: Used to increase receptive field without increasing parameters
4. **Global Average Pooling (GAP)**: Used instead of flattening layers before classification
5. **Total Receptive Field**: > 44 pixels
6. **Parameter Count**: < 200k parameters

## Model Architecture Details

The model follows a C1C2C3C40 architecture:
```python
1. Initial Conv Block (C1):
   - Conv2d(3, 32, 3) + BN + ReLU
   - Conv2d(32, 64, 3) + BN + ReLU

2. Depthwise Sep Conv Block (C2):
   - DepthwiseSeparableConv(64, 128, 3) + BN + ReLU
   - Conv2d(128, 128, 3) + BN + ReLU

3. Dilated Conv Block (C3):
   - Conv2d(128, 256, 3, dilation=2) + BN + ReLU
   - Conv2d(256, 256, 3) + BN + ReLU

4. Strided Conv Block (C4):
   - Conv2d(256, 512, 3, stride=2) + BN + ReLU

5. Output:
   - Global Average Pooling
   - FC(512, 10)
```

## Data Augmentation

Using the albumentations library with:
1. Horizontal Flip (p=0.5)
2. ShiftScaleRotate:
   - shift_limit=0.1
   - scale_limit=0.1
   - rotate_limit=15°
   - p=0.5
3. CoarseDropout:
   - max_holes=1
   - max_height=16px
   - max_width=16px
   - min_holes=1
   - min_height=16px
   - min_width=16px
   - fill_value=dataset_mean

## Project Structure

```
.
├── model.py          # Model architecture definition
├── dataset.py        # Dataset and DataLoader utilities
├── transforms.py     # Data augmentation transforms
├── utils.py          # Training utilities and metrics
├── train.py         # Training loop and functions
└── CIFAR10_Training.ipynb  # Jupyter notebook for training
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- albumentations
- numpy
- matplotlib
- tqdm

Install requirements:
```bash
pip install torch torchvision albumentations tqdm matplotlib numpy
```

## Training

The model can be trained either using the Python scripts or the Jupyter notebook:

1. Using Python scripts:
```bash
python train.py
```

2. Using Jupyter notebook:
```bash
jupyter notebook CIFAR10_Training.ipynb
```

## Model Performance

The model achieves the target accuracy of 85% on CIFAR-10 test set with:
- Training for 50 epochs
- Batch size of 128
- Adam optimizer with learning rate 0.001
- Cross Entropy Loss

## Training Logs

Example training output showing both training and validation metrics for each epoch:
```
 
Epoch: 1/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.47it/s, Loss=0.3099, Acc=89.17%]

Test set: Average loss: 0.4556, Accuracy: 84.47%


Epoch: 2/50
Training: 100%|██████████| 391/391 [01:51<00:00,  3.49it/s, Loss=0.2893, Acc=89.79%]

Test set: Average loss: 0.5348, Accuracy: 82.99%


Epoch: 3/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.48it/s, Loss=0.2779, Acc=90.22%]

Test set: Average loss: 0.4691, Accuracy: 84.62%


Epoch: 4/50
Training: 100%|██████████| 391/391 [01:51<00:00,  3.49it/s, Loss=0.2687, Acc=90.57%]

Test set: Average loss: 0.4573, Accuracy: 86.12%


Epoch: 5/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.49it/s, Loss=0.2572, Acc=91.01%]

Test set: Average loss: 0.4137, Accuracy: 86.77%


Epoch: 6/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.49it/s, Loss=0.2488, Acc=91.26%]

Test set: Average loss: 0.4471, Accuracy: 86.26%


Epoch: 7/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.49it/s, Loss=0.2394, Acc=91.63%]

Test set: Average loss: 0.4966, Accuracy: 84.75%


Epoch: 8/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.49it/s, Loss=0.2227, Acc=92.25%]

Test set: Average loss: 0.3981, Accuracy: 87.37%


Epoch: 9/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.49it/s, Loss=0.2178, Acc=92.33%]

Test set: Average loss: 0.4510, Accuracy: 86.30%


Epoch: 10/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.48it/s, Loss=0.2123, Acc=92.55%]

Test set: Average loss: 0.3978, Accuracy: 87.89%


Epoch: 11/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.47it/s, Loss=0.1998, Acc=93.03%]

Test set: Average loss: 0.3926, Accuracy: 88.25%


Epoch: 12/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.47it/s, Loss=0.1974, Acc=92.99%]

Test set: Average loss: 0.4579, Accuracy: 85.54%


Epoch: 13/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.48it/s, Loss=0.1909, Acc=93.29%]

Test set: Average loss: 0.4532, Accuracy: 87.41%


Epoch: 14/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.48it/s, Loss=0.1807, Acc=93.54%]

Test set: Average loss: 0.4227, Accuracy: 87.59%


Epoch: 15/50
Training: 100%|██████████| 391/391 [01:52<00:00,  3.49it/s, Loss=0.1829, Acc=93.55%]

Test set: Average loss: 0.4277, Accuracy: 87.62%

```

## Usage

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training:
```bash
python train.py
```

## Code Modularity

The code is organized into separate modules for better maintainability:
- `model.py`: Contains the model architecture
- `dataset.py`: Handles data loading and preprocessing
- `transforms.py`: Defines data augmentation transforms
- `utils.py`: Contains utility functions
- `train.py`: Implements the training loop

## License

MIT License
