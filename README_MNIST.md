# DRIFT MNIST Classifier

This project demonstrates the use of DRIFT (Data Reduction via Informative Feature Transformation) and PCA feature sets in a neural network classifier on the MNIST dataset.

## Features

- Generates custom mode shapes for DRIFT features
- Extracts DRIFT, PCA, and Full Model features from MNIST
- Trains and evaluates a configurable feedforward neural network on each feature set

## Files

- `drift_mnist.py` — Main code for generating features and training models.
- `requirements.txt` — Required Python packages.

## Usage

```bash
pip install -r requirements.txt
python drift_CIFAR100.py
the same for the plot .py file
```

## Configuration

Modify the `CONFIG` variable in `drift_CIFAR100.py` to change hidden layers, number of modes, epochs, etc.

## Requirements

- Python 3.8+
- numpy 1.23.5
- tensorflow 2.10.0
- scikit-learn 1.2.1
