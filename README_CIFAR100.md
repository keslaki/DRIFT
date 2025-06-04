# CIFAR-100 Feature Engineering and Neural Network Classification

This project demonstrates feature engineering techniques and neural network classification on the CIFAR-100 dataset. It explores three feature sets: **cosine similarity to analytic mode shapes**, **PCA features**, and **full scaled pixel data**. Each feature set is fed into a simple dense neural network for multi-class image classification.

## Table of Contents
- [Overview](#overview)
- [Features & Workflow](#features--workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [References](#references)

---

## Overview

- **Data**: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) image classification dataset.
- **Feature Extraction**:
  - Cosine similarity with analytic "mode shapes" (sinusoidal basis functions).
  - Principal Component Analysis (PCA) features.
  - Full scaled image data.
- **Model**: A fully-connected (dense) neural network with configurable hidden layers and activation functions.

---

## Features & Workflow

1. **Load CIFAR-100 images** and resize them to a configurable grid size (default: 32x32).
2. **Generate mode shapes** (sinusoidal 2D patterns, similar to eigenmodes in physics).
3. **Compute features**:
   - Cosine similarity between each image channel and each mode shape.
   - PCA on scaled/flattened image data.
   - Optionally, use the full scaled, flattened image pixels as features.
4. **Build and train a neural network** on each feature set.
5. **Evaluate** the model and record test accuracy.

---

## Installation

### Requirements

- Python 3.7+
- [TensorFlow](https://www.tensorflow.org/) (tested with 2.x)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)

Install requirements via pip:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## Usage

1. **Clone this repository** and copy the code into a Python script (e.g., `main.py`).

2. **Run the script**:
   ```bash
   python main.py
   ```

3. **Output**:
   - The script will train and evaluate a neural network on all three feature sets, printing validation accuracy for each.
   - Training histories for each model are stored in the `all_histories`, `all_accuracies`, and `all_names` lists.

---

## Configuration

Modify the `CONFIG` dictionary at the top of the script to control key parameters:

```python
CONFIG = {
    'grid_size': 32,                   # Image grid size (height/width)
    'num_modes': 80,                   # Number of analytic mode shapes per channel
    'num_classes': 100,                # CIFAR-100 classes
    'valida_split': 0.2,               # Validation split for training
    'nn_hidden_layers': [32],          # Hidden layers shape
    'epochs': 50,                      # Training epochs
    'batch_size': 2,                   # Batch size
    'activation_functions': ['relu'],  # Activation functions for NN
}
```

---

## Results

- Three feature sets are compared:
  1. **Cosine Similarity Features**: Project each channel onto analytic mode shapes.
  2. **PCA Features**: Principal components of the scaled image data.
  3. **Full Scaled Data**: All pixels after scaling and flattening.

- For each, a dense neural network is trained and evaluated. Typical output might look like:
  ```
  Feature set: Similarity Features (relu) - Test accuracy: 0.21
  Feature set: PCA Features (relu) - Test accuracy: 0.23
  Feature set: Full Scaled Data (relu) - Test accuracy: 0.25
  ```
  (Actual results will vary depending on configuration and compute resources.)

---

## References

- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [PCA in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [StandardScaler in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

## License

This project is for educational/research purposes.