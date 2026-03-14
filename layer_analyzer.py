import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

class LayerAnalyzer:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained(
            'gpt2',
            output_hidden_states=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def analyze_layers(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states

        layer_stats = []
        for layer_idx, hidden in enumerate(hidden_states):
            layer_data = hidden[0].numpy()
            stats = {
                'layer': layer_idx,
                'mean': float(np.mean(layer_data)),
                'std': float(np.std(layer_data)),
                'max': float(np.max(layer_data)),
                'min': float(np.min(layer_data)),
                'sparsity': float(
                    np.mean(np.abs(layer_data) < 0.1)
                ),
                'importance': float(
                    np.abs(layer_data).mean()
                )
            }
            layer_stats.append(stats)

        return layer_stats, tokens, hidden_states

    def visualize_layers(self, texts):
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(
            'NeuralLens — Layer Importance Analyzer\n'
            'Which Layers Matter Most?',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        all_results = []
        for text in texts:
            stats, tokens, hidden = self.analyze_layers(text)
            all_results.append({
                'text': text,
                'stats': stats,
                'tokens': tokens,
                'hidden': hidden
            })
            print(f"✅ Analyzed: '{text}'")

        # Plot 1 - Layer importance comparison
        ax1 = fig.add_subplot(3, 2, 1)
        colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']

        for idx, result in enumerate(all_results):
            layers = [s['layer'] for s in result['stats']]
            importance = [
                s['importance'] for s in result['stats']
            ]
            ax1.plot(
                layers,
                importance,
                'o-',
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"'{result['text'][:20]}'"
            )

        ax1.set_title(
            'Layer Importance Score',
            color='white',
            fontsize=11
        )
        ax1.set_xlabel('Layer', color='white')
        ax1.set_ylabel('Importance', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#16213e')
        ax1.legend(
            facecolor='#16213e',
            labelcolor='white',
            fontsize=8
        )
        for spine in ax1.spines.values():
            spine.set_edgecolor('#444')

        # Plot 2 - Layer sparsity
        ax2 = fig.add_subplot(3, 2, 2)
        for idx, result in enumerate(all_results):
            layers = [s['layer'] for s in result['stats']]
            sparsity = [
                s['sparsity'] for s in result['stats']
            ]
            ax2.plot(
                layers,
                sparsity,
                'o-',
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"'{result['text'][:20]}'"
            )

        ax2.set_title(
            'Layer Sparsity — How Many Neurons Are Silent',
            color='white',
            fontsize=11
        )
        ax2.set_xlabel('Layer', color='white')
        ax2.set_ylabel('Sparsity', color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#16213e')
        ax2.legend(
            facecolor='#16213e',
            labelcolor='white',
            fontsize=8
        )
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444')

        # Plot 3 - Layer std deviation
        ax3 = fig.add_subplot(3, 2, 3)
        for idx, result in enumerate(all_results):
            layers = [s['layer'] for s in result['stats']]
            std = [s['std'] for s in result['stats']]
            ax3.plot(
                layers,
                std,
                'o-',
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"'{result['text'][:20]}'"
            )

        ax3.set_title(
            'Layer Variance — Information Diversity',
            color='white',
            fontsize=11
        )
        ax3.set_xlabel('Layer', color='white')
        ax3.set_ylabel('Std Deviation', color='white')
        ax3.tick_params(colors='white')
        ax3.set_facecolor('#16213e')
        ax3.legend(
            facecolor='#16213e',
            labelcolor='white',
            fontsize=8
        )
        for spine in ax3.spines.values():
            spine.set_edgecolor('#444')

        # Plot 4 - Heatmap of layer activations
        ax4 = fig.add_subplot(3, 2, 4)
        importance_matrix = np.array([
            [s['importance'] for s in r['stats']]
            for r in all_results
        ])

        im = ax4.imshow(
            importance_matrix,
            aspect='auto',
            cmap='plasma'
        )
        ax4.set_title(
            'Layer Importance Heatmap',
            color='white',
            fontsize=11
        )
        ax4.set_xlabel('Layer', color='white')
        ax4.set_ylabel('Text', color='white')
        ax4.set_yticks(range(len(texts)))
        ax4.set_yticklabels(
            [t[:20] for t in texts],
            color='white',
            fontsize=8
        )
        ax4.tick_params(colors='white')
        ax4.set_facecolor('#16213e')
        plt.colorbar(im, ax=ax4)

        # Plot 5 - Most important layer per text
        ax5 = fig.add_subplot(3, 1, 3)
        for idx, result in enumerate(all_results):
            importance_scores = [
                s['importance'] for s in result['stats']
            ]
            most_important_layer = np.argmax(
                importance_scores
            )
            print(
                f"📊 '{result['text'][:20]}' → "
                f"Most important layer: {most_important_layer}"
            )

        layer_names = [f"L{i}" for i in range(13)]
        x = np.arange(len(layer_names))
        width = 0.25

        for idx, result in enumerate(all_results):
            importance = [
                s['importance'] for s in result['stats']
            ]
            ax5.bar(
                x + idx * width,
                importance,
                width,
                label=f"'{result['text'][:15]}'",
                color=colors[idx],
                alpha=0.8
            )

        ax5.set_title(
            'Layer by Layer Importance Comparison',
            color='white',
            fontsize=11
        )
        ax5.set_xlabel('Layer', color='white')
        ax5.set_ylabel('Importance Score', color='white')
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(
            layer_names,
            color='white'
        )
        ax5.tick_params(colors='white')
        ax5.set_facecolor('#16213e')
        ax5.legend(
            facecolor='#16213e',
            labelcolor='white'
        )
        for spine in ax5.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        plt.savefig(
            'layer_analysis.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("\n✅ Saved as 'layer_analysis.png'")
        plt.show()
        print("\n🎉 Layer analysis complete!")


# Run it
if __name__ == "__main__":
    analyzer = LayerAnalyzer()

    texts = [
        "The cat sat on the mat",
        "Paris is the capital of France",
        "Water is made of hydrogen and oxygen",
    ]

    analyzer.visualize_layers(texts)
