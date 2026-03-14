import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

class NeuronMonitor:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained(
            'gpt2',
            output_hidden_states=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def get_neuron_activations(self, text):
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
        return outputs.hidden_states, tokens

    def find_top_neurons(self, text, top_n=20):
        hidden_states, tokens = self.get_neuron_activations(text)

        # Analyze middle layer
        middle_layer = hidden_states[6][0]

        # Find most active neurons
        neuron_activations = middle_layer.abs().mean(dim=0)
        top_neurons = neuron_activations.topk(top_n)

        return {
            'indices': top_neurons.indices.tolist(),
            'values': top_neurons.values.tolist(),
            'tokens': tokens,
            'hidden_states': hidden_states
        }

    def visualize_neurons(self, texts):
        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(
            'NeuralLens — Neuron Activation Monitor\n'
            'Finding The Most Important Neurons',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        for idx, text in enumerate(texts):
            print(f"\n🔍 Analyzing: '{text}'")
            results = self.find_top_neurons(text)

            # Plot top neuron activations
            ax = fig.add_subplot(
                len(texts), 2,
                idx * 2 + 1
            )

            colors = plt.cm.plasma(
                np.array(results['values']) /
                max(results['values'])
            )

            bars = ax.bar(
                range(len(results['indices'])),
                results['values'],
                color=colors
            )

            ax.set_title(
                f'Top 20 Neurons — "{text[:30]}"',
                color='white',
                fontsize=9
            )
            ax.set_xlabel(
                'Neuron Rank',
                color='white',
                fontsize=8
            )
            ax.set_ylabel(
                'Activation',
                color='white',
                fontsize=8
            )
            ax.tick_params(colors='white')
            ax.set_facecolor('#16213e')

            # Add neuron indices
            ax.set_xticks(range(len(results['indices'])))
            ax.set_xticklabels(
                [f"N{n}" for n in results['indices']],
                rotation=90,
                fontsize=6,
                color='white'
            )

            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

            # Plot neuron activity across tokens
            ax2 = fig.add_subplot(
                len(texts), 2,
                idx * 2 + 2
            )

            # Get top 5 neurons across tokens
            hidden = results['hidden_states'][6][0]
            top_5_indices = results['indices'][:5]
            tokens = results['tokens']

            for neuron_idx in top_5_indices:
                activations = hidden[:, neuron_idx].numpy()
                ax2.plot(
                    range(len(tokens)),
                    activations,
                    'o-',
                    linewidth=2,
                    markersize=6,
                    label=f'N{neuron_idx}',
                    alpha=0.8
                )

            ax2.set_title(
                f'Top 5 Neurons Across Tokens',
                color='white',
                fontsize=9
            )
            ax2.set_xlabel(
                'Token Position',
                color='white',
                fontsize=8
            )
            ax2.set_ylabel(
                'Activation Value',
                color='white',
                fontsize=8
            )
            ax2.set_xticks(range(len(tokens)))
            ax2.set_xticklabels(
                tokens,
                rotation=45,
                ha='right',
                fontsize=7,
                color='white'
            )
            ax2.tick_params(colors='white')
            ax2.set_facecolor('#16213e')
            ax2.legend(
                facecolor='#16213e',
                labelcolor='white',
                fontsize=7
            )

            for spine in ax2.spines.values():
                spine.set_edgecolor('#444')

            print(
                f"✅ Top neurons: "
                f"{results['indices'][:5]}"
            )

        plt.tight_layout()
        plt.savefig(
            'neuron_monitor.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("\n✅ Saved as 'neuron_monitor.png'")
        plt.show()
        print("\n🎉 Neuron monitoring complete!")


# Run it
if __name__ == "__main__":
    monitor = NeuronMonitor()

    texts = [
        "The cat sat on the mat",
        "Paris is the capital of France",
        "Water is made of hydrogen and oxygen",
    ]

    monitor.visualize_neurons(texts)

