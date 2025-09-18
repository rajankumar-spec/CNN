# EVA4 Session 2 – CNN on MNIST  

## 📌 Overview  
This notebook implements a **Convolutional Neural Network (CNN)** for the MNIST digit classification task. The goal is to explore different convolutional and pooling layers, understand receptive fields, and optimize the architecture for accuracy and efficiency.  

---

## 🔄 Process  
1. **Data Preparation**
   - Dataset: **MNIST** handwritten digits (28×28 grayscale).
   - Normalization and standard transformations applied.  
   - Data loaders prepared with batching and shuffling.  

2. **Model Design**
   - Experimented with **3×3 convolutions**, **1×1 convolutions**, **max pooling**, **batch normalization**, and **dropout**.  
   - Used max pooling to both increase receptive field and reduce dimensionality.  
   - Applied 1×1 convolutions for feature mixing and controlling channel depth.  

3. **Training**
   - Loss function: **Negative Log Likelihood Loss (NLLLoss)**.  
   - Optimizer: **Stochastic Gradient Descent (SGD)** with momentum.  
   - Device: CUDA/GPU if available.  
   - Training loop tracks **losses** and **accuracy** for both training and test sets.  

---

## 🧠 Model Architecture  
The final CNN (`Net` class) consists of:  

- **Conv1**: 1 → 8 channels, 3×3 kernel, padding=1  
- **Conv2**: 8 → 8 channels, 3×3 kernel, padding=1  
- **MaxPool (2×2)**  
- **BatchNorm2d (8)**  
- **Conv3**: 8 → 16 channels, 3×3 kernel, padding=1  
- **Conv4**: 16 → 16 channels, 3×3 kernel  
- **MaxPool (2×2)**  
- **BatchNorm2d (16)**  
- **Dropout (0.1)**  
- **Conv5**: 16 → 32 channels, 3×3 kernel  
- **Conv6**: 32 → 32 channels, 3×3 kernel  
- **BatchNorm2d (32)**  
- **Fully Connected**: Linear(128 → 10)  
- **Output**: LogSoftmax  

---

## 📊 Model Parameters
![Model Summary](model_summary.png)  

---

## 📉 Training Losses
![Training Loss Curve](training_loss.png)  

---

## 📈 Test Accuracy
![Test Accuracy Curve](test_accuracy.png)  

---

## 🚀 How to Run  
1. Clone repo / open in Google Colab.  
2. Install dependencies:
   ```bash
   pip install torch torchvision torchsummary
3. Run the notebook cells sequentially.
4. Training progress and metrics will be logged.
   
