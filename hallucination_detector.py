import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

class HallucinationDetector:
    def __init__(self):
        print("🧠 Loading GPT2 model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Model loaded!")

    def calculate_confidence(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Calculate probability for each token
        probs = F.softmax(logits, dim=-1)

        token_confidences = []
        for i in range(len(tokens) - 1):
            # Probability the model assigned to actual next token
            next_token_id = inputs['input_ids'][0][i + 1]
            confidence = probs[0, i, next_token_id].item()
            token_confidences.append({
                'token': tokens[i + 1],
                'confidence': confidence,
                'is_hallucination': confidence < 0.1
            })

        return tokens, token_confidences

    def detect(self, text):
        print(f"\n🔍 Analyzing: '{text}'")
        tokens, confidences = self.calculate_confidence(text)

        # Overall hallucination risk
        avg_confidence = np.mean(
            [c['confidence'] for c in confidences]
        )
        low_confidence_count = sum(
            1 for c in confidences 
            if c['is_hallucination']
        )
        hallucination_risk = (
            low_confidence_count / len(confidences)
        ) * 100

        print(f"\n📊 Analysis Results:")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Hallucination Risk: {hallucination_risk:.1f}%")
        print(f"\n🔍 Token by Token Analysis:")
        for c in confidences:
            status = "⚠️ SUSPICIOUS" if c['is_hallucination'] else "✅ OK"
            print(
                f"   '{c['token']}' — "
                f"confidence: {c['confidence']:.3f} {status}"
            )

        return confidences, hallucination_risk, avg_confidence

    def visualize(self, texts):
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            'NeuralLens — Hallucination Detector',
            fontsize=16,
            fontweight='bold',
            color='white'
        )
        fig.patch.set_facecolor('#1a1a2e')

        for idx, text in enumerate(texts):
            confidences, risk, avg_conf = self.detect(text)

            ax = fig.add_subplot(len(texts), 1, idx + 1)

            tokens = [c['token'] for c in confidences]
            conf_values = [c['confidence'] for c in confidences]
            is_hallucination = [
                c['is_hallucination'] for c in confidences
            ]

            # Color bars based on confidence
            colors = [
                '#ff4444' if h else '#00ff88' 
                for h in is_hallucination
            ]

            bars = ax.bar(
                range(len(tokens)),
                conf_values,
                color=colors,
                alpha=0.8
            )

            # Add threshold line
            ax.axhline(
                y=0.1,
                color='yellow',
                linestyle='--',
                linewidth=2,
                label='Hallucination Threshold'
            )

            ax.set_title(
                f'"{text}" — Risk: {risk:.1f}% | '
                f'Avg Confidence: {avg_conf:.3f}',
                color='white',
                fontsize=10
            )
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(
                tokens,
                rotation=45,
                ha='right',
                color='white',
                fontsize=9
            )
            ax.set_ylabel('Confidence', color='white')
            ax.tick_params(colors='white')
            ax.set_facecolor('#16213e')
            ax.legend(
                facecolor='#16213e',
                labelcolor='white'
            )
            ax.set_ylim(0, 1)

            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        # Add legend
        red_patch = mpatches.Patch(
            color='#ff4444',
            label='⚠️ Suspicious — Low Confidence'
        )
        green_patch = mpatches.Patch(
            color='#00ff88',
            label='✅ Normal — High Confidence'
        )

        plt.tight_layout()
        plt.savefig(
            'hallucination_detection.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )
        print("\n✅ Saved as 'hallucination_detection.png'")
        plt.show()
        print("\n🎉 Hallucination detection complete!")


# Run it
if __name__ == "__main__":
    detector = HallucinationDetector()

    # Test with different texts
    # Some factual, some likely to cause hallucination
    texts = [
        "The capital of France is Paris",
        "The moon is made of cheese and candy",
        "Water is made of hydrogen and oxygen",
    ]

    import matplotlib.patches as mpatches
    detector.visualize(texts)
