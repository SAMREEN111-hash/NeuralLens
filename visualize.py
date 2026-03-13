import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from activation_extractor import ActivationExtractor

def visualize_activations(activations, text):
    
    print("Creating visualization...")
    
    # Setup the figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f'NeuralLens — Activation Map\nInput: "{text}"',
        fontsize=14,
        fontweight='bold',
        color='white'
    )
    fig.patch.set_facecolor('#1a1a2e')
    axes = axes.flatten()

    for idx, (layer_name, activation) in enumerate(activations.items()):
        if idx >= 12:
            break

        # Get activation data
        act_data = activation.numpy()

        # Plot heatmap for each layer
        im = axes[idx].imshow(
            act_data,
            aspect='auto',
            cmap='plasma'
        )

        axes[idx].set_title(
            f'Layer {idx}',
            color='white',
            fontsize=10
        )
        axes[idx].set_xlabel('Features', color='gray', fontsize=8)
        axes[idx].set_ylabel('Tokens', color='gray', fontsize=8)
        axes[idx].tick_params(colors='gray')
        axes[idx].set_facecolor('#16213e')

        for spine in axes[idx].spines.values():
            spine.set_edgecolor('#444')

        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig(
        'neurallens_activations.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#1a1a2e'
    )
    print("✅ Visualization saved as 'neurallens_activations.png'")
    plt.show()

# Run it
if __name__ == "__main__":
    text = "How does artificial intelligence work?"
    
    print("🧠 NeuralLens Visualization Starting...")
    print(f"📝 Analyzing: '{text}'")
    
    extractor = ActivationExtractor()
    activations = extractor.extract(text)
    
    visualize_activations(activations, text)
    
    print("\n🎉 Done! Check neurallens_activations.png")
