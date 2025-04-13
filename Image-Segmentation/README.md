# Image Segmentation Project: Mean-Shift & segnet-lite

## Overview

This project provides two image segmentation pipelines:

- **Mean-Shift Segmentation:**  
  Uses iterative mean-shift clustering to group pixels based on their color values after converting an image into the perceptually uniform CIELAB color space.

- **segnet-lite:**  
  Implements a lightweight version of the SegNet architecture, which performs end-to-end learning for semantic segmentation. The segnet-lite network is then used on a multi-digit MNIST dataset, where each pixel is labeled as background or one of the digit classes.

Both implementations are provided with full training, validation, and visualization pipelines,


## Content

- **Mean-Shift Segmentation:**
  - Full implementation of the algorithm from scratch.
  - vectorization of the operations for efficient computation.

- **segnet-lite:**
  - Lightweight network architecture with downsampling and upsampling paths.
  - Custom 2D cross-entropy loss function adapted for segmentation.
  - Training and validation scripts with logging and TensorBoard integration.
  - Dataset loader for multi-digit MNIST segmentation.
  - Visualization utilities to map segmentation labels to RGB colors.