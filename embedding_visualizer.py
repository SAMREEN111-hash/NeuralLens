import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class EmbeddingVisualizer:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def get_word_embeddings(self, words):
        embeddings = []
        valid_words = []

        for word in words:
            inputs = self.tokenizer(
                word,
                return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[0][0]
            embeddings.append(embedding.numpy())
            valid_words.append(word)
            print(f"✅ Got embedding for: '{word}'")

        return np.array(embeddings), valid_words

    def visualize_3d(self, word_groups):
        print("\n🔍 Creating 3D embedding space...")

        all_words = []
        all_embeddings = []
        all_groups = []
        all_colors = []

        colors = [
            '#ff6b6b', '#4ecdc4', '#ffe66d',
            '#a8e6cf', '#ff8b94', '#b8b8ff'
        ]

        for idx, (group_name, words) in enumerate(
            word_groups.items()
        ):
            embeddings, valid_words = \
                self.get_word_embeddings(words)
            all_words.extend(valid_words)
            all_embeddings.extend(embeddings)
            all_groups.extend(
                [group_name] * len(valid_words)
            )
            all_colors.extend(
                [colors[idx]] * len(valid_words)
            )

        all_embeddings = np.array(all_embeddings)

        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(all_embeddings)

        # Reduce to 2D as well
        pca_2d = PCA(n_components=2)
        coords_2d = pca_2d.fit_transform(all_embeddings)

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            'NeuralLens — Embedding Space Visualizer\n'
            'How AI Organizes Words In Its Mind',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        # Plot 1 - 3D visualization
        ax1 = fig.add_subplot(
            1, 2, 1,
            projection='3d'
        )
        ax1.set_facecolor('#16213e')

        unique_groups = list(word_groups.keys())
        for idx, group in enumerate(unique_groups):
            mask = [g == group for g in all_groups]
            group_coords = coords_3d[mask]

            ax1.scatter(
                group_coords[:, 0],
                group_coords[:, 1],
                group_coords[:, 2],
                c=colors[idx],
                label=group,
                s=200,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.5
            )

            # Add word labels
            words_in_group = [
                w for w, g in zip(all_words, all_groups)
                if g == group
            ]
            for i, word in enumerate(words_in_group):
                ax1.text(
                    group_coords[i, 0],
                    group_coords[i, 1],
                    group_coords[i, 2],
                    word,
                    fontsize=8,
                    color='white',
                    alpha=0.9
                )

        ax1.set_title(
            '3D Word Embedding Space',
            color='white',
            fontsize=12,
            pad=20
        )
        ax1.set_xlabel('PCA 1', color='white')
        ax1.set_ylabel('PCA 2', color='white')
        ax1.set_zlabel('PCA 3', color='white')
        ax1.tick_params(colors='white')
        ax1.legend(
            facecolor='#16213e',
            labelcolor='white',
            loc='upper left'
        )
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        # Plot 2 - 2D visualization
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_facecolor('#16213e')

        for idx, group in enumerate(unique_groups):
            mask = [g == group for g in all_groups]
            group_coords_2d = coords_2d[mask]

            ax2.scatter(
                group_coords_2d[:, 0],
                group_coords_2d[:, 1],
                c=colors[idx],
                label=group,
                s=300,
                alpha=0.8,
                edgecolors='white',
                linewidth=1
            )

            words_in_group = [
                w for w, g in zip(all_words, all_groups)
                if g == group
            ]
            for i, word in enumerate(words_in_group):
                ax2.annotate(
                    word,
                    (
                        group_coords_2d[i, 0],
                        group_coords_2d[i, 1]
                    ),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=9,
                    color='white',
                    fontweight='bold'
                )

        ax2.set_title(
            '2D Word Embedding Map',
            color='white',
            fontsize=12
        )
        ax2.set_xlabel('Dimension 1', color='white')
        ax2.set_ylabel('Dimension 2', color='white')
        ax2.tick_params(colors='white')
        ax2.legend(
            facecolor='#16213e',
            labelcolor='white'
        )
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        plt.savefig(
            'embedding_space.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("✅ Saved as 'embedding_space.png'")
        plt.show()
        print("\n🎉 Embedding visualization complete!")


# Run it
if __name__ == "__main__":
    visualizer = EmbeddingVisualizer()

    # Different categories of words
    word_groups = {
        "Animals": [
            "cat", "dog", "bird",
            "fish", "horse"
        ],
        "Countries": [
            "France", "Germany", "Japan",
            "Brazil", "India"
        ],
        "Science": [
            "atom", "quantum", "gravity",
            "energy", "molecule"
        ],
        "Emotions": [
            "happy", "sad", "angry",
            "excited", "calm"
        ],
    }

    visualizer.visualize_3d(word_groups)
