---
license: mit
task_categories:
- image-classification
tags:
- biology
- plants
- succulent
- unsupervised-learning
- dinov2
- ImageFolder
pretty_name: High-Quality Clustered Succulent Dataset
size_categories:
- 1K<n<10K
---


# Succulent Vision Dataset

## Introduction
This dataset contains **1,000 high-quality segmented images** of succulents. 
The original photos were taken in complex cluster environments, and individual succulent plants were precisely segmented using **SAM (Segment Anything Model)**.

## Processing Pipeline
To provide structured data, the images have been automatically categorized using an advanced unsupervised pipeline:
- **Feature Extraction**: [DINOv2](https://github.com/facebookresearch/dinov2) (Vision Transformer) for semantic shape features.
- **Color Analysis**: **HSV Histogram** analysis to prioritize botanical color variations.
- **Dimensionality Reduction**: **UMAP** for manifold learning.
- **Clustering**: **Agglomerative Hierarchical Clustering** to ensure morphological consistency.

## Dataset Structure
The dataset is organized into folders representing different species/morphological groups:
- `class_000/`: Echeveria-like structures (green)
- `class_001/`: Cabbage-like wrinkled leaves
- ... and more.

## Usage
Perfect for:
- Training **Conditional DDPM** or **GANs** for plant generation.
- Fine-tuning image classification models for botany.
- Testing unsupervised clustering algorithms.

**Maintained by**: [HaiPenglai](https://huggingface.co/HaiPenglai)