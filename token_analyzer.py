import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

class TokenAnalyzer:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def get_token_embeddings(self, text):
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )

        # Get token names
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # Get last layer hidden states
        last_hidden = outputs.last_hidden_state[0]

        return tokens, last_hidden

    def visualize_token_similarity(self, text):
        print(f"\n🔍 Analyzing tokens in: '{text}'")

        tokens, embeddings = self.get_token_embeddings(text)

        # Calculate similarity between all tokens
        emb_numpy = embeddings.numpy()
        similarity_matrix = cosine_similarity(emb_numpy)

        # Plot similarity matrix
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(
            f'NeuralLens — Token Relationship Map\n"{text}"',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        # Plot 1 - Similarity heatmap
        im1 = axes[0].imshow(
            similarity_matrix,
            cmap='RdYlGn',
            aspect='auto',
            vmin=-1,
            vmax=1
        )
        axes[0].set_title(
            'Token Similarity Matrix',
            color='white',
            fontsize=12
        )
        axes[0].set_xticks(range(len(tokens)))
        axes[0].set_yticks(range(len(tokens)))
        axes[0].set_xticklabels(
            tokens,
            rotation=45,
            ha='right',
            color='white',
            fontsize=10
        )
        axes[0].set_yticklabels(
            tokens,
            color='white',
            fontsize=10
        )
        axes[0].set_facecolor('#16213e')
        plt.colorbar(im1, ax=axes[0])

        # Add similarity values on heatmap
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                axes[0].text(
                    j, i,
                    f'{similarity_matrix[i,j]:.2f}',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8
                )

        # Plot 2 - Token activation strength
        activation_strength = np.abs(emb_numpy).mean(axis=1)
        colors = plt.cm.plasma(
            activation_strength / activation_strength.max()
        )

        bars = axes[1].bar(
            range(len(tokens)),
            activation_strength,
            color=colors
        )
        axes[1].set_title(
            'Token Activation Strength',
            color='white',
            fontsize=12
        )
        axes[1].set_xticks(range(len(tokens)))
        axes[1].set_xticklabels(
            tokens,
            rotation=45,
            ha='right',
            color='white',
            fontsize=10
        )
        axes[1].set_ylabel(
            'Average Activation',
            color='white'
        )
        axes[1].tick_params(colors='white')
        axes[1].set_facecolor('#16213e')
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        plt.savefig(
            'token_relationships.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("✅ Saved as 'token_relationships.png'")
        plt.show()
        print("\n🎉 Token analysis complete!")


# Run it
if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    text = "The cat sat on the mat"
