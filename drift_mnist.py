import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# Configuration
CONFIG = {
    'grid_size': 28,
    'num_modes': 50,
    'num_classes': 10,
    'valida_split': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'activation_functions': ['relu'],
    'hidden_layers': [64, 128, 64],
}

# Set seed
tf.random.set_seed(42)
np.random.seed(42)

def generate_nm_pairs(num_modes):
    side = int(np.ceil(np.sqrt(num_modes)))
    n, m = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1))
    return np.vstack([n.ravel(), m.ravel()]).T[:num_modes]

def generate_mode_shapes(grid_size, num_modes, Lx, Ly):
    start_time = time.time()
    print('Generating Mode Shapes...')
    x = np.linspace(0, Lx, grid_size)
    y = np.linspace(0, Ly, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    pairs = generate_nm_pairs(num_modes)
    modes_2d = np.zeros((num_modes, grid_size, grid_size))
    for i, (m, n) in enumerate(pairs):
        modes_2d[i] = np.sin(m * np.pi * X_grid / Lx) * np.sin(n * np.pi * Y_grid / Ly)
    modes_flat = modes_2d.reshape(num_modes, -1)
    print(f"Mode shape generation completed in {time.time() - start_time:.2f} seconds.")
    return modes_flat

def load_preprocess_mnist():
    start_time = time.time()
    print("Loading and preprocessing MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    y_train_one_hot = to_categorical(y_train, CONFIG['num_classes'])
    y_test_one_hot = to_categorical(y_test, CONFIG['num_classes'])
    print(f"Data loaded in {time.time() - start_time:.2f} seconds.")
    return x_train_flat, x_test_flat, y_train_one_hot, y_test_one_hot, y_train, y_test

def compute_features(x_train_flat, x_test_flat, modes_flat, num_modes):
    start_time = time.time()
    print(f"Calculating {num_modes} DRIFT features...")
    x_train_drift = cosine_similarity(x_train_flat, modes_flat)
    x_test_drift = cosine_similarity(x_test_flat, modes_flat)
    print(f"DRIFT features in {time.time() - start_time:.2f} seconds.")
    start_time = time.time()
    print("Calculating PCA features...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    pca = PCA(n_components=num_modes)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    print(f"PCA features in {time.time() - start_time:.2f} seconds.")
    return x_train_drift, x_test_drift, x_train_scaled, x_test_scaled, x_train_pca, x_test_pca


def build_nn_model(input_shape, activation='relu'):
    model = Sequential()
    # The first layer needs input_shape to know input dimension
    model.add(Dense(CONFIG['hidden_layers'][0], activation=activation, input_shape=(input_shape,)))
    # Add remaining layers except the first one
    for size in CONFIG['hidden_layers'][1:]:
        model.add(Dense(size, activation=activation))
    model.add(Dense(CONFIG['num_classes'], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def train_evaluate_model(model, x_train, x_test, y_train, y_test, y_test_labels, name, activation):
    start_time = time.time()
    print(f"Training {name} ({activation})...")
    history = model.fit(
        x_train, y_train,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_split=CONFIG['valida_split'],
        verbose=1,
        shuffle=False
    )
    print(f"Trained {name} in {time.time() - start_time:.2f} seconds.")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, batch_size=CONFIG['batch_size'], verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    top1_accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(f"{name} ({activation}) - Test accuracy: {accuracy:.4f}, Top-1: {top1_accuracy:.4f}")
    return history, accuracy, top1_accuracy

def main():
    """Run all steps and prepare data for plots."""
    # Generate mode shapes
    modes_flat = generate_mode_shapes(
        CONFIG['grid_size'], CONFIG['num_modes'], CONFIG['grid_size'], CONFIG['grid_size']
    )
    # Load data
    x_train_flat, x_test_flat, y_train_one_hot, y_test_one_hot, y_train, y_test = load_preprocess_mnist()
    # Compute features
    x_train_drift, x_test_drift, x_train_scaled, x_test_scaled, x_train_pca, x_test_pca = compute_features(
        x_train_flat, x_test_flat, modes_flat, CONFIG['num_modes']
    )
    feature_sets = [
        (x_train_drift, x_test_drift, "DRIFT Features"),
        (x_train_pca, x_test_pca, "PCA Features"),
        (x_train_scaled, x_test_scaled, "Full Scaled Data")
    ]
    results = []
    histories = []

    for activation in CONFIG['activation_functions']:
        for x_train, x_test, name in feature_sets:
            model = build_nn_model(x_train.shape[1], activation=activation)
            history, accuracy, top1_accuracy = train_evaluate_model(
                model, x_train, x_test, y_train_one_hot, y_test_one_hot, y_test, name, activation
            )
            results.append({'name': name, 'activation': activation, 'accuracy': accuracy, 'top1': top1_accuracy})
            histories.append({'name': name, 'activation': activation, 'history': history})

    return histories  # return histories for plotting

if __name__ == "__main__":
    histories = main()