# DRIFT MNIST Classifier

This project demonstrates the use of DRIFT (Discrete Fourier Transform-inspired) and PCA feature sets in a neural network classifier on the MNIST dataset.

## Features

- Generates custom mode shapes for DRIFT features
- Extracts DRIFT, PCA, and scaled features from MNIST
- Trains and evaluates a configurable feedforward neural network on each feature set

## Files

- `drift_mnist.py` — Main code for generating features and training models.
- `requirements.txt` — Required Python packages.

## Usage

```bash
pip install -r requirements.txt
python drift_mnist.py
```

## Configuration

Modify the `CONFIG` variable in `drift_mnist.py` to change hidden layers, number of modes, epochs, etc.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- numpy
