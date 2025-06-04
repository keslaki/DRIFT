import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load results saved from training
with open('training_results.pkl', 'rb') as f:
    data = pickle.load(f)

histories = data['histories']  # list of dicts containing training histories
names = data['names']          # list of model names
accuracies = data.get('accuracies', None)  # optional

# Use seaborn palette for colors (or fallback)
default_colors = sns.color_palette("tab10")
colors =   ['blue', 'red', 'black']

def plot_accuracy(histories, legend_names=None):
    plt.figure(figsize=(6, 4))
    for idx, history in enumerate(histories):
        label_base = legend_names[idx] if legend_names and idx < len(legend_names) else names[idx]
        color = colors[idx % len(colors)]
        plt.plot(history['accuracy'], label=f'{label_base} Train', color=color, linestyle='-')
        plt.plot(history['val_accuracy'], label=f'{label_base} Val', color=color, linestyle='--')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig('training_validation_accuracy.png', dpi=300)
    plt.show()

def plot_loss(histories, legend_names=None):
    plt.figure(figsize=(6, 4))
    for idx, history in enumerate(histories):
        label_base = legend_names[idx] if legend_names and idx < len(legend_names) else names[idx]
        color = colors[idx % len(colors)]
        plt.plot(history['loss'], label=f'{label_base} Train', color=color, linestyle='-')
        plt.plot(history['val_loss'], label=f'{label_base} Val', color=color, linestyle='--')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('training_validation_loss.png', dpi=300)
    plt.show()

# Example usage with optional custom legend names:
custom_legends = ['DRFIT', 'PCA', 'Full Model']  # or None to use saved names
plot_accuracy(histories, legend_names=custom_legends)
plot_loss(histories, legend_names=custom_legends)
