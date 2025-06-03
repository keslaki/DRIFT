import matplotlib.pyplot as plt

def plot_accuracy(histories, legend_names=None):
    plt.figure(figsize=(6, 4))
    colors = ['blue', 'red', 'black']
    for idx, item in enumerate(histories):
        history = item['history']
        # Use provided legend name or default
        label_base = legend_names[idx] if legend_names and idx < len(legend_names) else f"{item['name']} ({item['activation']})"
        color = colors[idx % len(colors)]
        plt.plot(history.history['accuracy'], label=f'{label_base} Train', color=color, linestyle='-')
        plt.plot(history.history['val_accuracy'], label=f'{label_base} Val', color=color, linestyle='--')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=300)
    plt.show()

def plot_loss(histories, legend_names=None):
    plt.figure(figsize=(6, 4))
    colors = ['blue', 'red', 'black']
    for idx, item in enumerate(histories):
        history = item['history']
        label_base = legend_names[idx] if legend_names and idx < len(legend_names) else f"{item['name']} ({item['activation']})"
        color = colors[idx % len(colors)]
        plt.plot(history.history['loss'], label=f'{label_base} Train', color=color, linestyle='-')
        plt.plot(history.history['val_loss'], label=f'{label_base} Val', color=color, linestyle='--')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=300)
    plt.show()

    
custom_legends = ['DRIFT', 'PCA', 'Full Model']
plot_accuracy(histories, legend_names=custom_legends)
plot_loss(histories, legend_names=custom_legends)
