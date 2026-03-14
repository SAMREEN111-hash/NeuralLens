import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

class FeatureAttribution:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def compute_gradients(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )

        # Get embeddings with gradients
        input_ids = inputs['input_ids']
        
        # Get embedding layer
        embeddings = self.model.transformer.wte(input_ids)
        embeddings.retain_grad()
        embeddings_with_grad = embeddings.clone().detach()
        embeddings_with_grad.requires_grad_(True)

        # Forward pass
        outputs = self.model(
            inputs_embeds=embeddings_with_grad
        )
        
        # Get prediction for last token
        logits = outputs.logits[0, -1, :]
        top_token = logits.argmax()
        
        # Backward pass
        logits[top_token].backward()
        
        # Get gradients
        gradients = embeddings_with_grad.grad[0]
        
        # Calculate attribution scores
        attribution = (
            gradients * embeddings_with_grad[0]
        ).sum(dim=-1).abs().detach().numpy()

        predicted_token = self.tokenizer.decode(top_token)

        return tokens, attribution, predicted_token

    def visualize_attribution(self, texts):
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            'NeuralLens — Feature Attribution Analyzer\n'
            'Which Words Influenced The AI Most?',
            fontsize=14,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        for idx, text in enumerate(texts):
            print(f"\n🔍 Analyzing: '{text}'")

            try:
                tokens, attribution, predicted = \
                    self.compute_gradients(text)

                ax = fig.add_subplot(
                    len(texts), 1, idx + 1
                )

                # Normalize attribution scores
                attribution_norm = (
                    attribution / attribution.max()
                )

                # Color based on importance
                colors = plt.cm.RdYlGn(attribution_norm)

                bars = ax.bar(
                    range(len(tokens)),
                    attribution_norm,
                    color=colors,
                    alpha=0.9
                )

                ax.set_title(
                    f'"{text}" → AI predicts: '
                    f'"{predicted}"',
                    color='white',
                    fontsize=10
                )
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(
                    tokens,
                    rotation=45,
                    ha='right',
                    color='white',
                    fontsize=10
                )
                ax.set_ylabel(
                    'Attribution Score',
                    color='white'
                )
                ax.tick_params(colors='white')
                ax.set_facecolor('#16213e')
                ax.set_ylim(0, 1.2)

                # Add value labels
                for bar, val in zip(bars, attribution_norm):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.02,
                        f'{val:.2f}',
                        ha='center',
                        va='bottom',
                        color='white',
                        fontsize=8
                    )

                for spine in ax.spines.values():
                    spine.set_edgecolor('#444')

                print(
                    f"✅ Predicted next token: "
                    f"'{predicted}'"
                )
                print(
                    f"📊 Most influential token: "
                    f"'{tokens[attribution.argmax()]}'"
                )

            except Exception as e:
                print(f"⚠️ Error: {e}")

        plt.tight_layout()
        plt.savefig(
            'feature_attribution.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("\n✅ Saved as 'feature_attribution.png'")
        plt.show()
        print("\n🎉 Feature attribution complete!")


# Run it
if __name__ == "__main__":
    analyzer = FeatureAttribution()

    texts = [
        "The capital of France is",
        "The cat sat on the",
        "Water is made of hydrogen and",
        "The largest planet in our solar system is",
    ]

    analyzer.visualize_attribution(texts)
