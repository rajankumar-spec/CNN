MNIST Digit Classification (PyTorch)
📌 Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.
The notebook (Copy_of_ERA_Session_4.ipynb) walks through dataset preparation, model building, training, and evaluation.

⚙️ Requirements

Python 3.8+

PyTorch

Torchvision

Matplotlib

NumPy

tqdm

Install dependencies:

pip install torch torchvision matplotlib numpy tqdm

📂 Dataset

Dataset: MNIST

Training samples: 60,000

Test samples: 10,000

Image size: 28 × 28 grayscale

Data is automatically downloaded via torchvision.datasets.

🏗 Model Architecture

The model is a CNN with ~23K parameters:

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
            Conv2d-2           [-1, 16, 24, 24]           1,168
            Conv2d-3           [-1, 64, 10, 10]           9,280
            Conv2d-4             [-1, 16, 8, 8]           9,232
            Conv2d-5              [-1, 8, 2, 2]           1,160
            Linear-6                   [-1, 64]           2,112
            Linear-7                   [-1, 10]             650
================================================================
Total params: 23,682
Trainable params: 23,682
Non-trainable params: 0
----------------------------------------------------------------

🚀 Training

Optimizer: SGD (assumed default)

Batch size: 16

Device: CPU (CUDA not available)

Epochs: 5

Training Logs (sample)
Epoch 1
Test set: Average loss: 0.0068, Accuracy: 96.79%

Epoch 3
Test set: Average loss: 0.0049, Accuracy: 97.64%

Epoch 5
Test set: Average loss: 0.0047, Accuracy: 97.69%


✅ Final test accuracy: ~97.7%

📊 Results
Accuracy & Loss Curves

Sample Predictions

▶️ Usage

To run the notebook:

jupyter notebook Copy_of_ERA_Session_4.ipynb
