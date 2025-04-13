# Image Classification with PyTorch

## Overview
This notebook implements image classification on the MNIST and CIFAR-10 datasets using PyTorch. It includes training and evaluation of both a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN).

## steps
- Data preprocessing, normalization, and augmentation
- Implementation of MLP and CNN architectures
- Training and evaluation on MNIST and CIFAR-10
- hyperparameter tuning to achieve higher test accuracy

## Model Architectures
### MLP
- Fully connected layers with ReLU activations
- Optimized using Adam optimizer
- Tuned hyperparameters for accuracy improvement

### CNN
| Layer          | Parameters                                  |
|---------------|--------------------------------------------|
| Conv Layer 1  | 1 input channel, 32 output channels, 3x3 kernel |
| ReLU          | -                                          |
| Conv Layer 2  | 32 input channels, 64 output channels, 3x3 kernel |
| ReLU          | -                                          |
| MaxPooling    | 2x2 kernel                                 |
| Dropout       | p=0.25                                     |
| Fully Connected | 9216 input, 128 output neurons           |
| ReLU          | -                                          |
| Dropout       | p=0.5                                      |
| Fully Connected | 128 input, 10 output (classification)    |

## Results
- **99%+** accuracy on MNIST with CNN.
- **46%** accuracy on CIFAR-10 with CNN.