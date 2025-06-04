🧠 CIFAR-100 Feature Comparison with Cosine Similarity, PCA, and Full Features
This repository evaluates different feature extraction strategies—Cosine Similarity with predefined basis functions, Principal Component Analysis (PCA), and Full Flattened Data—for training a neural network on the CIFAR-100 dataset. Results are saved for later visualization and analysis.

📦 Features
Cosine Similarity: Projects image data onto sinusoidal mode shapes.

PCA: Reduces dimensionality of full image data.

Full Flattened Features: Uses all pixel values as input after standard scaling.

Model Training: Each feature set is used to train a feedforward neural network.

Comparison: Results (accuracy/loss) across strategies are saved for plotting.

🧠 Model Configuration
Dataset: CIFAR-100

Architecture: Fully Connected Neural Network with 3 hidden layers

Framework: TensorFlow / Keras

Activation: relu (configurable)

Output: Classification into 100 categories

🛠️ How It Works
Mode Shape Generation
Generates sinusoidal basis functions over a 2D grid to be used for cosine similarity projection.

Data Preparation
Loads CIFAR-100, resizes images, flattens channels, and creates one-hot labels.

Feature Engineering

Cosine similarity with mode shapes

PCA on scaled pixel data

Flattened & scaled raw data

Training
Each model is trained and evaluated separately using:

python
Copy
Edit
model.fit(..., validation_split=0.2, epochs=50, batch_size=128)
Saving Results
Stores all training histories, model names, and test accuracies in a single training_results.pkl file.

📊 Output
After training, the script saves:

Copy
Edit
training_results.pkl
Containing:

histories: Training/validation loss and accuracy for each model.

accuracies: Final test accuracy.

names: Model identifiers, e.g., "PCA Features (relu)".

You can later use the plotting script (visualize_training.py, not included here) to generate performance comparison plots.

🚀 Quick Start
bash
Copy
Edit
pip install tensorflow matplotlib numpy pandas scikit-learn
python train_feature_models.py
This will execute the entire pipeline and save the results to training_results.pkl.

📁 Folder Structure (Suggested)
bash
Copy
Edit
.
├── train_feature_models.py         # Main training pipeline
├── training_results.pkl            # Saved training histories and results
├── visualize_training.py           # Optional: script to visualize training results
├── README.md
📌 Configuration
Edit the CONFIG dictionary at the top of the script to adjust:

grid_size: Grid resolution for mode shapes and image resizing

num_modes: Number of sinusoidal basis functions

nn_hidden_layers: Neural network architecture

activation_functions: One or more activation functions to try

🧪 Example Use Cases
Compare engineered vs. learned features

Benchmark sinusoidal features against PCA or raw pixels

Visualize feature effectiveness through training metrics

📬 Feedback & Contributions
Feel free to open issues or contribute improvements. Ideas for feature augmentation, deeper networks, or performance visualization are welcome!
