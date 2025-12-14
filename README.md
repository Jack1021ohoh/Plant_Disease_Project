# Plant Disease Detection using Vision Transformers

A deep learning project that leverages Vision Transformer (ViT) models to classify plant diseases in apple trees. This system achieves 81.4% F1-score on the validation set using advanced data augmentation and focal loss optimization.

## Overview

This project implements a multi-label classification system for identifying five types of apple tree diseases:
- Powdery Mildew
- Scab
- Complex (multiple diseases)
- Frog Eye Leaf Spot
- Rust

The model uses a Vision Transformer (`vit_small_patch16_224`) pre-trained on ImageNet-21k and fine-tuned on the Plant Pathology 2021 FGVC8 dataset.

## Key Features

- **Vision Transformer Architecture**: Utilizes state-of-the-art transformer-based image classification
- **Focal Loss**: Custom implementation to handle class imbalance with weighted loss computation
- **Data Augmentation**: Comprehensive augmentation pipeline including color jitter, perspective transforms, and affine transformations
- **Multi-label Classification**: Handles images with multiple simultaneous disease labels
- **Reproducible Results**: Fixed random seeds across all libraries for consistent experimentation

## Model Performance

- **Training F1-Score**: 81.96%
- **Validation F1-Score**: 81.42%
- **Training Loss**: 0.0615
- **Validation Loss**: 0.0704

## Technologies Used

- **Deep Learning Framework**: PyTorch
- **Model Library**: timm (PyTorch Image Models)
- **Data Processing**: NumPy, Pandas, PIL
- **Machine Learning**: scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development Platform**: Kaggle

## Dataset

**Source**: [Plant Pathology 2021 - FGVC8](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8)

The dataset contains images of apple tree leaves with various disease conditions. Each image can have multiple labels or be labeled as healthy.

## Architecture Details

### Model Configuration
- **Base Model**: Vision Transformer Small (vit_small_patch16_224.augreg_in21k)
- **Input Size**: 224x224 pixels
- **Number of Classes**: 5 (multi-label output)
- **Optimizer**: Adam with cosine annealing scheduler
- **Loss Function**: Focal Loss (α=1.0, γ=2.0) with class weights

### Training Configuration
- **Epochs**: 15
- **Batch Size**: 64
- **Initial Learning Rate**: 1e-3
- **Minimum Learning Rate**: 1e-6
- **Train/Validation Split**: 80/20

### Data Augmentation Pipeline
- Random color jitter (brightness, contrast, saturation)
- Random perspective transforms
- Random affine transformations
- Random horizontal and vertical flips
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Project Structure

```
Plant_Disease_Project/
├── plant-disease-model.ipynb    # Main training and inference notebook
└── README.md                     # Project documentation
```

## Implementation Highlights

1. **Custom Focal Loss**: Addresses class imbalance by down-weighting well-classified examples and focusing on hard negatives
2. **Class Weighting**: Individual weights for each disease class to handle imbalanced distribution
3. **Reproducibility**: Comprehensive seed setting across Python, NumPy, and PyTorch
4. **Multi-label Handling**: Binary cross-entropy setup for simultaneous disease detection

## Results

The model demonstrates strong performance in detecting multiple plant diseases simultaneously, with validation F1-score improving from 30.68% (epoch 1) to 81.42% (epoch 15). The focal loss implementation with class weights proved effective in handling the imbalanced nature of disease occurrence.

## About

This project was developed as part of the Advanced Data Science with IBM course, demonstrating practical application of modern deep learning techniques in agricultural disease detection.

## License

This project uses data from the [Plant Pathology 2021 FGVC8](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8) Kaggle competition.
