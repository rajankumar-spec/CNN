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



---
