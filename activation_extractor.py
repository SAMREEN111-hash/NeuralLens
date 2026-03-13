import torch
from transformers import GPT2Model, GPT2Tokenizer

class ActivationExtractor:
    def __init__(self):
        print("Loading GPT2 model...")
        self.model = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.activations = {}
        print("Model loaded successfully!")
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output[0].detach()
        return hook
    
    def extract(self, text):
        # Clear previous activations
        self.activations = {}
        
        # Register hooks on each layer
        hooks = []
        for i, layer in enumerate(self.model.h):
            hook = layer.register_forward_hook(
                self.get_activation(f'layer_{i}')
            )
            hooks.append(hook)
        
        # Process input text
        inputs = self.tokenizer(
            text, 
            return_tensors='pt'
        )
        
        # Run the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Remove hooks after use
        for hook in hooks:
            hook.remove()
            
        return self.activations

# Test it immediately
if __name__ == "__main__":
    extractor = ActivationExtractor()
    
    text = "How does artificial intelligence work?"
    print(f"\nAnalyzing: '{text}'")
    
    activations = extractor.extract(text)
    
    print(f"\n✅ Successfully extracted activations!")
    print(f"📊 Number of layers captured: {len(activations)}")
    print(f"\nLayer Details:")
    for layer_name, activation in activations.items():
        print(f"  {layer_name}: shape {activation.shape}")
