import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Config
CONFIG = {
    'grid_size': 32,
    'num_modes': 40,
    'num_classes': 100,
    'valida_split': 0.2,
    'nn_hidden_layers': [64, 128, 64],
    'epochs': 50,
    'batch_size': 128,
    'activation_functions': ['relu'],
}

tf.random.set_seed(42)
np.random.seed(42)

def generate_nm_pairs(num_modes):
    side = int(np.ceil(np.sqrt(num_modes)))
    n, m = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1))
    return np.vstack([n.ravel(), m.ravel()]).T[:num_modes]

def generate_mode_shapes(grid_size, num_modes, Lx, Ly):
    x = np.linspace(0, Lx, grid_size)
    y = np.linspace(0, Ly, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    pairs = generate_nm_pairs(num_modes)
    modes_2d = np.zeros((num_modes * 3, grid_size, grid_size))
    for i, (m, n) in enumerate(pairs):
        mode_shape = np.sin(m * np.pi * X_grid / Lx) * np.sin(n * np.pi * Y_grid / Ly)
        modes_2d[i] = mode_shape
        modes_2d[i + num_modes] = mode_shape
        modes_2d[i + 2 * num_modes] = mode_shape
    modes_flat = modes_2d.reshape(num_modes * 3, -1)
    return modes_flat, modes_2d, X_grid, Y_grid

def resize_dataset(images, size):
    images = tf.convert_to_tensor(images, dtype=tf.float32) / 255.0
    resized_images = tf.image.resize(images, size, method='bilinear', antialias=True)
    resized_images_uint8 = np.clip(resized_images.numpy() * 255, 0, 255).astype(np.uint8)
    return resized_images_uint8

def load_preprocess_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train_resized = resize_dataset(x_train, [CONFIG['grid_size'], CONFIG['grid_size']])
    x_test_resized = resize_dataset(x_test, [CONFIG['grid_size'], CONFIG['grid_size']])
    x_train_flat = x_train_resized.reshape(x_train_resized.shape[0], -1)
    x_test_flat = x_test_resized.reshape(x_test_resized.shape[0], -1)
    y_train_one_hot = to_categorical(y_train, CONFIG['num_classes'])
    y_test_one_hot = to_categorical(y_test, CONFIG['num_classes'])
    return x_train_resized, x_test_resized, x_train_flat, x_test_flat, y_train_one_hot, y_test_one_hot

def compute_cosine_similarity_features(x_data, modes_flat, num_modes):
    x_sim = []
    for i in range(len(x_data)):
        sample_img = x_data[i]
        feats = []
        for c in range(3):
            channel_flat = sample_img[:, :, c].flatten()
            for basis in modes_flat[c * num_modes:(c + 1) * num_modes]:
                sim = np.dot(channel_flat, basis) / (np.linalg.norm(channel_flat) * np.linalg.norm(basis))
                feats.append(sim)
        x_sim.append(feats)
    return np.array(x_sim)

def compute_features(x_train, x_test, x_train_flat, x_test_flat, modes_flat, num_modes):
    x_train_sim = compute_cosine_similarity_features(x_train, modes_flat, num_modes)
    x_test_sim = compute_cosine_similarity_features(x_test, modes_flat, num_modes)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    pca = PCA(n_components=num_modes * 3)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    return x_train_sim, x_test_sim, x_train_scaled, x_test_scaled, x_train_pca, x_test_pca

def build_nn_model(input_shape, activation='relu'):
    model = Sequential()
    model.add(Dense(CONFIG['nn_hidden_layers'][0], activation=activation, input_shape=(input_shape,)))
    for layer_size in CONFIG['nn_hidden_layers'][1:]:
        model.add(Dense(layer_size, activation=activation))
    model.add(Dense(CONFIG['num_classes'], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_evaluate_model(model, x_train, x_test, y_train, y_test):
    history = model.fit(x_train, y_train,
                        epochs=CONFIG['epochs'],
                        batch_size=CONFIG['batch_size'],
                        validation_split=CONFIG['valida_split'],
                        verbose=1,
                        shuffle=False)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return history, accuracy

# === RUN THIS TO TRAIN ===

modes_flat, modes_2d, X_grid, Y_grid = generate_mode_shapes(CONFIG['grid_size'], CONFIG['num_modes'], CONFIG['grid_size'], CONFIG['grid_size'])
x_train, x_test, x_train_flat, x_test_flat, y_train_one_hot, y_test_one_hot = load_preprocess_cifar100()
x_train_sim, x_test_sim, x_train_scaled, x_test_scaled, x_train_pca, x_test_pca = compute_features(x_train, x_test, x_train_flat, x_test_flat, modes_flat, CONFIG['num_modes'])

feature_sets = [
    (x_train_sim, x_test_sim, "Similarity Features"),
    (x_train_pca, x_test_pca, "PCA Features"),
    (x_train_scaled, x_test_scaled, "Full Scaled Data")
]

# After training each model
all_histories = []
all_accuracies = []
all_names = []

for activation in CONFIG['activation_functions']:
    for x_tr, x_te, name in feature_sets:
        model = build_nn_model(x_tr.shape[1], activation=activation)
        history, accuracy = train_evaluate_model(model, x_tr, x_te, y_train_one_hot, y_test_one_hot)
        all_histories.append(history.history)  # Save only history dict
        all_accuracies.append(accuracy)
        all_names.append(f"{name} ({activation})")

# Save histories, accuracies, and names for later plotting
import pickle
with open('training_results.pkl', 'wb') as f:
    pickle.dump({
        'histories': all_histories,
        'accuracies': all_accuracies,
        'names': all_names
    }, f)
print("Training complete and results saved.")
