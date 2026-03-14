import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class CircuitAnalyzer:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained(
            'gpt2',
            output_hidden_states=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def get_neuron_activations(self, texts):
        all_activations = []
        all_labels = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )

            # Get middle layer activations
            middle_layer = outputs.hidden_states[6]
            # Mean across tokens
            mean_activation = middle_layer[0].mean(dim=0)
            all_activations.append(
                mean_activation.numpy()
            )
            all_labels.append(text[:20])

        return np.array(all_activations), all_labels

    def find_circuits(self, activations, n_clusters=3):
        # Use PCA to reduce dimensions
        pca = PCA(n_components=5)
        reduced = pca.fit_transform(activations)

        # Find neuron clusters
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        clusters = kmeans.fit_predict(reduced)

        return clusters, reduced, pca

    def visualize_circuits(self, texts):
        print(f"\n🔬 Analyzing neural circuits...")
        print(f"📝 Processing {len(texts)} texts...")

        activations, labels = self.get_neuron_activations(texts)

        clusters, reduced, pca = self.find_circuits(
            activations,
            n_clusters=3
        )

        # Further reduce to 2D for visualization
        pca_2d = PCA(n_components=2)
        coords_2d = pca_2d.fit_transform(activations)

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            'NeuralLens — Neural Circuit Analyzer\n'
            'Discovering What Neurons Have Learned',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        # Plot 1 - Circuit clusters in 2D
        ax1 = fig.add_subplot(2, 2, 1)
        colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']
        cluster_names = [
            'Circuit A',
            'Circuit B',
            'Circuit C'
        ]

        for cluster_id in range(3):
            mask = clusters == cluster_id
            ax1.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=colors[cluster_id],
                label=cluster_names[cluster_id],
                s=200,
                alpha=0.8,
                edgecolors='white',
                linewidth=1
            )

        # Add text labels
        for i, label in enumerate(labels):
            ax1.annotate(
                label,
                (coords_2d[i, 0], coords_2d[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color='white',
                alpha=0.8
            )

        ax1.set_title(
            'Neural Circuits — Concept Clusters',
            color='white',
            fontsize=11
        )
        ax1.set_xlabel('Circuit Dimension 1', color='white')
        ax1.set_ylabel('Circuit Dimension 2', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#16213e')
        ax1.legend(
            facecolor='#16213e',
            labelcolor='white'
        )
        for spine in ax1.spines.values():
            spine.set_edgecolor('#444')

        # Plot 2 - Activation heatmap
        ax2 = fig.add_subplot(2, 2, 2)
        im = ax2.imshow(
            activations[:, :50],
            aspect='auto',
            cmap='plasma'
        )
        ax2.set_title(
            'Neuron Activation Patterns',
            color='white',
            fontsize=11
        )
        ax2.set_xlabel('Neuron Index', color='white')
        ax2.set_ylabel('Text Sample', color='white')
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(
            labels,
            fontsize=7,
            color='white'
        )
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#16213e')
        plt.colorbar(im, ax=ax2)

        # Plot 3 - Circuit assignment
        ax3 = fig.add_subplot(2, 1, 2)
        circuit_colors = [
            colors[c] for c in clusters
        ]
        bars = ax3.barh(
            range(len(labels)),
            [1] * len(labels),
            color=circuit_colors,
            alpha=0.8
        )

        ax3.set_title(
            'Circuit Assignment Per Text',
            color='white',
            fontsize=11
        )
        ax3.set_yticks(range(len(labels)))
        ax3.set_yticklabels(
            labels,
            color='white',
            fontsize=9
        )
        ax3.tick_params(colors='white')
        ax3.set_facecolor('#16213e')
        ax3.set_xlabel('Circuit Type', color='white')

        # Add circuit labels
        for i, cluster_id in enumerate(clusters):
            ax3.text(
                0.5, i,
                cluster_names[cluster_id],
                ha='center',
                va='center',
                color='white',
                fontweight='bold',
                fontsize=9
            )

        for spine in ax3.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        plt.savefig(
            'neural_circuits.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("✅ Saved as 'neural_circuits.png'")
        plt.show()
        print("\n🎉 Circuit analysis complete!")
        print("\n📊 Circuit Assignments:")
        for i, (label, cluster) in enumerate(
            zip(labels, clusters)
        ):
            print(
                f"   '{label}' → "
                f"{cluster_names[cluster]}"
            )


# Run it
if __name__ == "__main__":
    analyzer = CircuitAnalyzer()

    # Different types of text to find circuits
    texts = [
        "The cat sat on the mat",
        "Paris is the capital of France",
        "Water is H2O molecule",
        "The dog ran across the field",
        "London is in United Kingdom",
        "Oxygen has atomic number 8",
        "The bird flew over the house",
        "Berlin is capital of Germany",
        "Carbon dioxide is CO2 gas",
    ]

    analyzer.visualize_circuits(texts)
