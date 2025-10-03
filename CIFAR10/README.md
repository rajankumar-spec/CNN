# EVA4 Session 7– CNN on CIFAR10 

## 📌 Overview  
This notebook implements a **Convolutional Neural Network (CNN)** for the CIFAR10 classification task. The goal is to explore different convolutional and pooling layers, understand receptive fields, and optimize the architecture for accuracy and efficiency.  

---

## 🔄 Process  
1. **Data augmentation**
   - horizontal flip
   - shiftScaleRotate
   - coarseDropout
   - Normalization and standard transformations applied.  
   - Data loaders prepared with batching and shuffling.

2. **Model Design**
   - Experimented with **3×3 convolutions**, **Depthwise Separable Convolution**, **Dilated Convolution**, **batch normalization**, and **dropout**.  
   - Used GAP to both increase receptive field and reduce dimensionality. 

3. **Training**
   - Loss function: **Negative Log Likelihood Loss (NLLLoss)**.  
   - Optimizer: **Stochastic Gradient Descent (SGD)** with momentum.  
   - Device: CUDA/GPU if available.  
   - Training loop tracks **losses** and **accuracy** for both training and test sets.  

---
---



## 📊 Model Parameters
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #     
================================================================

            Conv2d-1           [-1, 32, 30, 30]             896
              ReLU-2           [-1, 32, 30, 30]               0
       BatchNorm2d-3           [-1, 32, 30, 30]              64
            Conv2d-4           [-1, 32, 26, 26]           9,248
              ReLU-5           [-1, 32, 26, 26]               0
       BatchNorm2d-6           [-1, 32, 26, 26]              64
            Conv2d-7           [-1, 32, 22, 22]           9,248
              ReLU-8           [-1, 32, 22, 22]               0
       BatchNorm2d-9           [-1, 32, 22, 22]              64
          Dropout-10           [-1, 32, 22, 22]               0
           Conv2d-11           [-1, 64, 20, 20]           9,280
             ReLU-12           [-1, 64, 20, 20]               0
      BatchNorm2d-13           [-1, 64, 20, 20]             128
           Conv2d-14           [-1, 64, 16, 16]          36,928
             ReLU-15           [-1, 64, 16, 16]               0
      BatchNorm2d-16           [-1, 64, 16, 16]             128
           Conv2d-17           [-1, 64, 12, 12]          36,928
             ReLU-18           [-1, 64, 12, 12]               0
      BatchNorm2d-19           [-1, 64, 12, 12]             128
          Dropout-20           [-1, 64, 12, 12]               0
           Conv2d-21            [-1, 128, 8, 8]          73,856
             ReLU-22            [-1, 128, 8, 8]               0
      BatchNorm2d-23            [-1, 128, 8, 8]             256
          Dropout-24            [-1, 128, 8, 8]               0
AdaptiveAvgPool2d-25            [-1, 128, 1, 1]               0
           Linear-26                   [-1, 10]           1,290
           
================================================================

Total params: 178,506
Trainable params: 178,506
Non-trainable params: 0

----------------------------------------------------------------

Input size (MB): 0.01
Forward/backward pass size (MB): 3.12
Params size (MB): 0.68
Estimated Total Size (MB): 3.81

----------------------------------------------------------------


---

## 📉 Test Losses

Epoch 1

Train: Loss=1.3961 Batch_id=781 Accuracy=44.87: 100%|██████████| 782/782 [00:24<00:00, 31.50it/s]

Test set: Average loss: 0.0193, Accuracy: 27915/50000 (55.83%)

Epoch 2

Train: Loss=0.9033 Batch_id=781 Accuracy=57.64: 100%|██████████| 782/782 [00:24<00:00, 32.08it/s]

Test set: Average loss: 0.0166, Accuracy: 30914/50000 (61.83%)

Epoch 3

Train: Loss=1.4903 Batch_id=781 Accuracy=63.03: 100%|██████████| 782/782 [00:23<00:00, 32.59it/s]

Test set: Average loss: 0.0147, Accuracy: 33234/50000 (66.47%)

Epoch 4

Train: Loss=0.9302 Batch_id=781 Accuracy=66.60: 100%|██████████| 782/782 [00:24<00:00, 32.33it/s]

Test set: Average loss: 0.0130, Accuracy: 35343/50000 (70.69%)

Epoch 5

Train: Loss=1.2249 Batch_id=781 Accuracy=69.60: 100%|██████████| 782/782 [00:24<00:00, 32.22it/s]
Test set: Average loss: 0.0122, Accuracy: 36194/50000 (72.39%)

Epoch 6

Train: Loss=0.8106 Batch_id=781 Accuracy=71.30: 100%|██████████| 782/782 [00:23<00:00, 32.84it/s]

Test set: Average loss: 0.0111, Accuracy: 37551/50000 (75.10%)

Epoch 7

Train: Loss=0.9940 Batch_id=781 Accuracy=72.98: 100%|██████████| 782/782 [00:25<00:00, 30.56it/s]

Test set: Average loss: 0.0108, Accuracy: 37989/50000 (75.98%)

Epoch 8

Train: Loss=1.2756 Batch_id=781 Accuracy=74.41: 100%|██████████| 782/782 [00:24<00:00, 32.51it/s]

Test set: Average loss: 0.0102, Accuracy: 38533/50000 (77.07%)

Epoch 9

Train: Loss=0.8267 Batch_id=781 Accuracy=75.83: 100%|██████████| 782/782 [00:24<00:00, 31.53it/s]

Test set: Average loss: 0.0098, Accuracy: 38909/50000 (77.82%)

Epoch 10

Train: Loss=0.6396 Batch_id=781 Accuracy=76.84: 100%|██████████| 782/782 [00:24<00:00, 31.85it/s]

Test set: Average loss: 0.0094, Accuracy: 39375/50000 (78.75%)

Epoch 11

Train: Loss=0.2316 Batch_id=781 Accuracy=79.94: 100%|██████████| 782/782 [00:25<00:00, 31.27it/s]

Test set: Average loss: 0.0080, Accuracy: 41097/50000 (82.19%)

Epoch 12

Train: Loss=0.6724 Batch_id=781 Accuracy=80.52: 100%|██████████| 782/782 [00:25<00:00, 30.55it/s]

Test set: Average loss: 0.0077, Accuracy: 41530/50000 (83.06%)

Epoch 13

Train: Loss=0.2238 Batch_id=781 Accuracy=80.96: 100%|██████████| 782/782 [00:25<00:00, 30.24it/s]

Test set: Average loss: 0.0076, Accuracy: 41630/50000 (83.26%)

Epoch 14

Train: Loss=0.8650 Batch_id=781 Accuracy=81.36: 100%|██████████| 782/782 [00:25<00:00, 31.06it/s]

Test set: Average loss: 0.0072, Accuracy: 42045/50000 (84.09%)

Epoch 15

Train: Loss=0.9739 Batch_id=781 Accuracy=81.70: 100%|██████████| 782/782 [00:25<00:00, 30.51it/s]

Test set: Average loss: 0.0072, Accuracy: 42020/50000 (84.04%)

Epoch 16

Train: Loss=0.7435 Batch_id=781 Accuracy=82.25: 100%|██████████| 782/782 [00:24<00:00, 31.32it/s]

Test set: Average loss: 0.0069, Accuracy: 42347/50000 (84.69%)

Epoch 17

Train: Loss=0.3695 Batch_id=781 Accuracy=82.55: 100%|██████████| 782/782 [00:25<00:00, 30.91it/s]

Test set: Average loss: 0.0068, Accuracy: 42485/50000 (84.97%)

Epoch 18

Train: Loss=0.4701 Batch_id=781 Accuracy=83.12: 100%|██████████| 782/782 [00:25<00:00, 31.12it/s]

Test set: Average loss: 0.0066, Accuracy: 42718/50000 (85.44%)

Epoch 19

Train: Loss=0.4444 Batch_id=781 Accuracy=83.15: 100%|██████████| 782/782 [00:24<00:00, 32.23it/s]

Test set: Average loss: 0.0064, Accuracy: 42960/50000 (85.92%)

Epoch 20

Train: Loss=0.4146 Batch_id=781 Accuracy=83.35: 100%|██████████| 782/782 [00:24<00:00, 32.46it/s]

Test set: Average loss: 0.0063, Accuracy: 43099/50000 (86.20%)

Epoch 21

Train: Loss=0.2305 Batch_id=781 Accuracy=85.12: 100%|██████████| 782/782 [00:23<00:00, 32.88it/s]

Test set: Average loss: 0.0056, Accuracy: 43915/50000 (87.83%)

Epoch 22

Train: Loss=1.1567 Batch_id=781 Accuracy=85.76: 100%|██████████| 782/782 [00:24<00:00, 32.35it/s]

Test set: Average loss: 0.0055, Accuracy: 44106/50000 (88.21%)

Epoch 23

Train: Loss=0.1696 Batch_id=781 Accuracy=85.83: 100%|██████████| 782/782 [00:24<00:00, 32.45it/s]

Test set: Average loss: 0.0054, Accuracy: 44116/50000 (88.23%)

Epoch 24

Train: Loss=0.4186 Batch_id=781 Accuracy=86.12: 100%|██████████| 782/782 [00:24<00:00, 32.36it/s]

Test set: Average loss: 0.0053, Accuracy: 44280/50000 (88.56%)

Epoch 25

Train: Loss=0.5100 Batch_id=781 Accuracy=86.35: 100%|██████████| 782/782 [00:24<00:00, 31.78it/s]

Test set: Average loss: 0.0053, Accuracy: 44281/50000 (88.56%)

---
