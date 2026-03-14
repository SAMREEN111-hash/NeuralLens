import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

class ModelComparator:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained(
            'gpt2',
            output_hidden_states=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def get_text_representation(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )
        # Get representation from each layer
        layer_representations = []
        for hidden in outputs.hidden_states:
            rep = hidden[0].mean(dim=0).numpy()
            layer_representations.append(rep)
        return layer_representations

    def compare_texts(self, text_pairs):
        print("\n🔍 Comparing texts across all layers...")
        results = []
        for text1, text2 in text_pairs:
            rep1 = self.get_text_representation(text1)
            rep2 = self.get_text_representation(text2)

            # Calculate similarity at each layer
            layer_similarities = []
            for l1, l2 in zip(rep1, rep2):
                sim = cosine_similarity(
                    l1.reshape(1, -1),
                    l2.reshape(1, -1)
                )[0][0]
                layer_similarities.append(sim)

            results.append({
                'text1': text1,
                'text2': text2,
                'similarities': layer_similarities
            })
            print(f"✅ Compared: '{text1[:20]}' vs '{text2[:20]}'")

        return results

    def visualize_comparison(self, text_pairs):
        results = self.compare_texts(text_pairs)

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            'NeuralLens — Model Comparison Tool\n'
            'How Similarly Does AI Process Different Texts?',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        # Plot 1 - Layer similarity curves
        ax1 = fig.add_subplot(2, 1, 1)
        colors = [
            '#ff6b6b', '#4ecdc4', '#ffe66d',
            '#a8e6cf', '#ff8b94'
        ]

        for idx, result in enumerate(results):
            layers = range(len(result['similarities']))
            label = (
                f"'{result['text1'][:15]}' vs "
                f"'{result['text2'][:15]}'"
            )
            ax1.plot(
                layers,
                result['similarities'],
                'o-',
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=6,
                label=label
            )

        ax1.set_title(
            'Similarity Across Layers',
            color='white',
            fontsize=12
        )
        ax1.set_xlabel('Layer', color='white')
        ax1.set_ylabel('Cosine Similarity', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#16213e')
        ax1.legend(
            facecolor='#16213e',
            labelcolor='white',
            fontsize=8
        )
        ax1.set_ylim(0, 1)
        ax1.axhline(
            y=0.9,
            color='yellow',
            linestyle='--',
            alpha=0.5,
            label='High Similarity'
        )
        for spine in ax1.spines.values():
            spine.set_edgecolor('#444')

        # Plot 2 - Final similarity comparison
        ax2 = fig.add_subplot(2, 1, 2)
        final_similarities = [
            r['similarities'][-1] for r in results
        ]
        labels = [
            f"'{r['text1'][:12]}'\nvs\n'{r['text2'][:12]}'"
            for r in results
        ]

        bar_colors = [
            '#00ff88' if s > 0.9
            else '#ff8b00' if s > 0.7
            else '#ff4444'
            for s in final_similarities
        ]

        bars = ax2.bar(
            range(len(labels)),
            final_similarities,
            color=bar_colors,
            alpha=0.8
        )

        ax2.set_title(
            'Final Layer Similarity Score',
            color='white',
            fontsize=12
        )
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(
            labels,
            color='white',
            fontsize=8
        )
        ax2.set_ylabel('Similarity Score', color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#16213e')
        ax2.set_ylim(0, 1)

        # Add value labels on bars
        for bar, val in zip(bars, final_similarities):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                color='white',
                fontsize=10,
                fontweight='bold'
            )

        # Add legend
        green = plt.Rectangle(
            (0, 0), 1, 1,
            fc='#00ff88'
        )
        orange = plt.Rectangle(
            (0, 0), 1, 1,
            fc='#ff8b00'
        )
        red = plt.Rectangle(
            (0, 0), 1, 1,
            fc='#ff4444'
        )
        ax2.legend(
            [green, orange, red],
            ['High Similar (>0.9)',
             'Medium Similar (>0.7)',
             'Low Similar (<0.7)'],
            facecolor='#16213e',
            labelcolor='white'
        )

        for spine in ax2.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        plt.savefig(
            'model_comparison.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("\n✅ Saved as 'model_comparison.png'")
        plt.show()
        print("\n🎉 Comparison complete!")
        print("\n📊 Final Similarity Scores:")
        for result in results:
            score = result['similarities'][-1]
            status = (
                "🟢 Very Similar" if score > 0.9
                else "🟡 Somewhat Similar" if score > 0.7
                else "🔴 Very Different"
            )
            print(
                f"   '{result['text1'][:20]}' vs "
                f"'{result['text2'][:20]}'"
                f" → {score:.3f} {status}"
            )


# Run it
if __name__ == "__main__":
    comparator = ModelComparator()

    # Compare different text pairs
    text_pairs = [
        (
            "The cat sat on mat",
            "The dog sat on floor"
        ),
        (
            "Paris is in France",
            "London is in England"
        ),
        (
            "I love pizza",
            "Quantum physics is complex"
        ),
        (
            "Water is H2O",
            "Water is a liquid"
        ),
        (
            "The sun is hot",
            "Ice cream is cold"
        ),
    ]

    comparator.visualize_comparison(text_pairs)
