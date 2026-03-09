
# MNIST Neural Network From Scratch

This project implements a fully connected neural network **from scratch using NumPy** to classify handwritten digits from the MNIST dataset.

Unlike most implementations that rely on frameworks such as PyTorch or TensorFlow, this project manually implements the core components of neural networks including forward propagation, softmax classification, cross-entropy loss, backpropagation, and mini-batch gradient descent.

The model achieves **~97% test accuracy** on the MNIST dataset.

---

# Project Overview

This project demonstrates how neural networks work internally by implementing the full training pipeline without using machine learning libraries.

Core components implemented manually:

- Forward propagation
- Sigmoid activation
- Softmax output layer
- Cross-entropy loss
- Backpropagation
- Mini-batch gradient descent
- Weight initialization
- Dataset standardization

The goal of this project is to understand neural network mechanics **at a mathematical and algorithmic level**.

---

# Model Architecture

Input layer  
784 features (28×28 grayscale pixels)

Hidden layer  
1200 neurons using sigmoid activation

Output layer  
10 neurons (digits 0–9) using softmax

---

# Hyperparameters

Hidden layer size: **1200**  
Epochs: **100**  
Batch size: **600**  
Learning rate: **0.8**

Final test accuracy: **~97%**

---

# Dataset

The model is trained on the **MNIST handwritten digit dataset**, which contains:

- 60,000 training images
- 10,000 test images
- 28×28 grayscale pixels per image
- 10 classification labels (digits 0–9)

Dataset source:

http://yann.lecun.com/exdb/mnist/

Download the following files and place them in the `data/` folder:

train-images-idx3-ubyte.gz  
train-labels-idx1-ubyte.gz  
t10k-images-idx3-ubyte.gz  
t10k-labels-idx1-ubyte.gz  

---

# Project Structure

mnist-neural-network-from-scratch  
│  
├── data  
│   └── MNIST dataset files  
│  
├── src  
│   ├── dataset.py  
│   ├── neural_network.py  
│   └── train.py  
│  
├── README.md  
├── requirements.txt  
└── .gitignore  

---

# Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/mnist-neural-network-from-scratch.git  

Navigate to the project directory:

cd mnist-neural-network-from-scratch  

Install dependencies:

pip install -r requirements.txt  

---

# Running the Project

Run the training script:

python src/train.py  

During training the program prints the loss and test accuracy for each batch so you can monitor the learning progress.

Example output:

0-0 > Loss: 2.30, Accuracy: 10.00%  
0-1 > Loss: 2.10, Accuracy: 18.50%  

---

# Key Learning Outcomes

This project demonstrates understanding of:

- Neural network fundamentals
- Backpropagation and gradient descent
- Multi-class classification with softmax
- Data preprocessing and normalization
- Implementing machine learning algorithms without frameworks

---

# Future Improvements

Possible extensions include:

- ReLU activation
- Deeper networks
- Adam optimizer
- Visualization of learned weights
- GPU implementation with PyTorch

---

# Author

Irtaza Abbasi  
Computer Engineering Student  
Toronto Metropolitan University
