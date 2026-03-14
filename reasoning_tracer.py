import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ReasoningTracer:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2LMHeadModel.from_pretrained(
            'gpt2',
            output_attentions=True,
            output_hidden_states=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def trace_reasoning(self, prompt):
        print(f"\n🔍 Tracing reasoning for: '{prompt}'")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        )
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )

        # Run model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get hidden states from all layers
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        # Calculate importance score per layer
        layer_importance = []
        for layer_idx, hidden in enumerate(hidden_states):
            # Mean activation magnitude per layer
            importance = hidden[0].abs().mean().item()
            layer_importance.append(importance)

        # Find most important token per layer
        token_importance_per_layer = []
        for layer_idx, hidden in enumerate(hidden_states):
            token_scores = hidden[0].abs().mean(dim=1)
            most_important = token_scores.argmax().item()
            token_importance_per_layer.append({
                'layer': layer_idx,
                'token': tokens[most_important],
                'score': token_scores.max().item()
            })

        # Get predicted next token
        logits = outputs.logits[0, -1, :]
        top_5_ids = logits.topk(5).indices
        top_5_tokens = [
            self.tokenizer.decode(id) 
            for id in top_5_ids
        ]
        top_5_scores = logits.topk(5).values.tolist()

        return {
            'tokens': tokens,
            'layer_importance': layer_importance,
            'token_importance_per_layer': token_importance_per_layer,
            'top_predictions': list(zip(top_5_tokens, top_5_scores))
        }

    def visualize_reasoning(self, prompt):
        results = self.trace_reasoning(prompt)

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            f'NeuralLens — Reasoning Path Tracer\n"{prompt}"',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        # Plot 1 - Layer importance path
        ax1 = fig.add_subplot(2, 2, 1)
        layers = range(len(results['layer_importance']))
        importance = results['layer_importance']

        ax1.plot(
            layers,
            importance,
            'o-',
            color='#00ff88',
            linewidth=2,
            markersize=8
        )
        ax1.fill_between(
            layers,
            importance,
            alpha=0.3,
            color='#00ff88'
        )
        ax1.set_title(
            'Reasoning Path — Layer Importance',
            color='white',
            fontsize=11
        )
        ax1.set_xlabel('Layer', color='white')
        ax1.set_ylabel('Importance Score', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#16213e')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#444')

        # Plot 2 - Most important token per layer
        ax2 = fig.add_subplot(2, 2, 2)
        layer_nums = [
            d['layer'] 
            for d in results['token_importance_per_layer']
        ]
        scores = [
            d['score'] 
            for d in results['token_importance_per_layer']
        ]
        token_labels = [
            d['token'] 
            for d in results['token_importance_per_layer']
        ]

        colors = plt.cm.plasma(
            np.array(scores) / max(scores)
        )
        bars = ax2.bar(layer_nums, scores, color=colors)

        ax2.set_title(
            'Most Active Token Per Layer',
            color='white',
            fontsize=11
        )
        ax2.set_xlabel('Layer', color='white')
        ax2.set_ylabel('Activation Score', color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#16213e')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444')

        # Add token labels on bars
        for bar, label in zip(bars, token_labels):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                label,
                ha='center',
                va='bottom',
                color='white',
                fontsize=8,
                rotation=45
            )

        # Plot 3 - Next token predictions
        ax3 = fig.add_subplot(2, 1, 2)
        pred_tokens = [p[0] for p in results['top_predictions']]
        pred_scores = [p[1] for p in results['top_predictions']]

        colors2 = plt.cm.RdYlGn(
            np.linspace(0.3, 1, len(pred_tokens))
        )
        bars2 = ax3.barh(
            pred_tokens,
            pred_scores,
            color=colors2
        )

        ax3.set_title(
            "NeuralLens Prediction — What AI Thinks Comes Next",
            color='white',
            fontsize=11
        )
        ax3.set_xlabel('Confidence Score', color='white')
        ax3.tick_params(colors='white')
        ax3.set_facecolor('#16213e')
        for spine in ax3.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        plt.savefig(
            'reasoning_path.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("✅ Saved as 'reasoning_path.png'")
        plt.show()
        print("\n🎉 Reasoning trace complete!")
        print(f"\n🔮 Top predictions after '{prompt}':")
        for token, score in results['top_predictions']:
            print(f"   '{token}' — score: {score:.2f}")


# Run it
if __name__ == "__main__":
    tracer = ReasoningTracer()
    tracer.visualize_reasoning("The capital of France is")
