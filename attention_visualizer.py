import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

class AttentionVisualizer:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained(
            'gpt2',
            output_attentions=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def get_attentions(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )

        # Get token words for labels
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )

        # Run model and get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract attention from all layers
        attentions = outputs.attentions

        return attentions, tokens

    def visualize_layer(self, attentions, tokens, layer=0):
        print(f"\n📊 Visualizing Layer {layer} Attention...")

        # Get attention for specific layer
        # Shape: [heads, tokens, tokens]
        layer_attention = attentions[layer][0]

        num_heads = layer_attention.shape[0]

        # Create grid of attention heads
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(
            f'NeuralLens — Attention Patterns\nLayer {layer} — All 12 Attention Heads',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')
        axes = axes.flatten()

        for head in range(num_heads):
            # Get attention weights for this head
            attn = layer_attention[head].numpy()

            # Plot heatmap
            im = axes[head].imshow(
                attn,
                cmap='hot',
                aspect='auto',
                vmin=0,
                vmax=1
            )

            axes[head].set_title(
                f'Head {head}',
                color='white',
                fontsize=10
            )

            # Set token labels
            axes[head].set_xticks(range(len(tokens)))
            axes[head].set_yticks(range(len(tokens)))
            axes[head].set_xticklabels(
                tokens,
                rotation=45,
                ha='right',
                fontsize=7,
                color='white'
            )
            axes[head].set_yticklabels(
                tokens,
                fontsize=7,
                color='white'
            )
            axes[head].set_facecolor('#16213e')

            plt.colorbar(im, ax=axes[head])

        plt.tight_layout()
        filename = f'attention_layer_{layer}.png'
        plt.savefig(
            filename,
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print(f"✅ Saved as '{filename}'")
        plt.show()

    def visualize_all_layers(self, text):
        print(f"\n🔍 Analyzing: '{text}'")
        attentions, tokens = self.get_attentions(text)

        print(f"✅ Got {len(attentions)} layers")
        print(f"✅ Tokens: {tokens}")
        print(f"✅ Each layer has {attentions[0].shape[1]} heads")

        # Visualize first 3 layers
        for layer in range(3):
            self.visualize_layer(attentions, tokens, layer)

        print("\n🎉 Attention visualization complete!")


# Run it
if __name__ == "__main__":
    visualizer = AttentionVisualizer()
    text = "The cat sat on the mat"
    visualizer.visualize_all_layers(text)
