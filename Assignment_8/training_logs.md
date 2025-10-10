# Training Logs - ResNet18 on CIFAR-100

## Training Parameters
- Model: ResNet18
- Dataset: CIFAR-100
- Batch Size: 128
- Initial Learning Rate: 0.1
- Optimizer: SGD with momentum (0.9)
- Weight Decay: 5e-4
- Learning Rate Scheduler: OneCycleLR

## Training Progress

### Early Training Phase (Epochs 0-10)
- Epoch 0: Train Loss: 3.822, Train Acc: 11.412% | Test Loss: 3.357, Test Acc: 19.270%
- Epoch 5: Train Loss: 1.999, Train Acc: 45.632% | Test Loss: 2.089, Test Acc: 45.360%
- Epoch 10: Train Loss: 1.519, Train Acc: 56.988% | Test Loss: 1.873, Test Acc: 51.520%

### Mid Training Phase (Epochs 25-50)
- Epoch 25: Train Loss: 1.174, Train Acc: 65.950% | Test Loss: 1.469, Test Acc: 59.770%
- Epoch 33: Train Loss: 1.090, Train Acc: 68.186% | Test Loss: 1.469, Test Acc: 60.690%
- Epoch 41: Train Loss: 1.030, Train Acc: 69.804% | Test Loss: 1.428, Test Acc: 61.110%
- Epoch 50: Train Loss: 0.952, Train Acc: 71.974% | Test Loss: 1.439, Test Acc: 62.440%

### Late Training Phase (Epochs 75-99)
- Epoch 75: Train Loss: 0.555, Train Acc: 82.804% | Test Loss: 1.344, Test Acc: 65.580%
- Epoch 85: Train Loss: 0.195, Train Acc: 94.284% | Test Loss: 1.169, Test Acc: 71.470%
- Epoch 90: Train Loss: 0.068, Train Acc: 98.518% | Test Loss: 1.030, Test Acc: 74.690%
- Epoch 95: Train Loss: 0.034, Train Acc: 99.514% | Test Loss: 0.984, Test Acc: 75.650%
- Epoch 99: Train Loss: 0.029, Train Acc: 99.650% | Test Loss: 0.982, Test Acc: 75.340%

## Key Milestones
1. First 50% Test Accuracy: Achieved at Epoch 8 (50.360%)
2. First 60% Test Accuracy: Achieved at Epoch 33 (60.690%)
3. First 70% Test Accuracy: Achieved at Epoch 83 (70.830%)
4. Peak Test Accuracy: 75.650% (Epoch 95)

## Training Observations

1. Initial Learning Phase (Epochs 0-10):
   - Rapid improvement from 19.27% to 51.52% test accuracy
   - Fast decrease in training loss from 3.822 to 1.519

2. Middle Phase (Epochs 11-50):
   - Steady improvement in test accuracy from 51% to 62%
   - Gradual reduction in training loss
   - Learning rate adjustments helping maintain progress

3. Final Phase (Epochs 51-99):
   - Training accuracy approaches 100% (99.650%)
   - Test accuracy improves from 62% to 75.65%
   - Model shows signs of overfitting but still improves test performance

4. Performance Analysis:
   - Exceeded target accuracy of 73% (reached 75.65%)
   - Training accuracy reached near-perfect levels
   - Gap between train and test accuracy indicates some overfitting
   - Loss values decreased consistently throughout training

## Hardware Performance
- Average epoch time: ~45 seconds
- Training iterations per second: ~9 it/s
- Testing iterations per second: ~30 it/s
- Total training time: ~75 minutes

## Final Results
- Best Test Accuracy: 75.650% (Epoch 95)
- Final Test Accuracy: 75.340% (Epoch 99)
- Final Training Accuracy: 99.650%
- Final Training Loss: 0.029
- Final Test Loss: 0.982