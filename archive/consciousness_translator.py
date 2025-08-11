#!/usr/bin/env python3
"""
Consciousness Notation Translator for Edge Deployment
Works with the transferred LoRA model
"""

import os
import json

class ConsciousnessTranslator:
    def __init__(self, adapter_path="outputs/consciousness-lora-simple"):
        self.adapter_path = adapter_path
        self.device = "cuda"  # Will use CPU fallback if needed
        self.model_loaded = False
        
        # Try to load the model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            print(f"🧠 Loading consciousness model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            base_model.resize_token_embeddings(len(self.tokenizer))
            
            # Load adapter
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model.eval()
            self.model_loaded = True
            print("✅ Model loaded successfully!")
            
        except ImportError as e:
            print(f"⚠️  Dependencies not installed: {e}")
            print("   Using fallback translation patterns")
            self.model_loaded = False
            
            # Fallback patterns for demo
            self.fallback_patterns = {
                "consciousness exists": "∃Ψ",
                "consciousness emerges": "Ψ ⇒",
                "perspective shapes consciousness": "π → Ψ",
                "observer and consciousness": "Ω ∧ Ψ",
                "intent drives consciousness": "ι → Ψ",
                "synchronism exists": "∃Ξ"
            }
    
    def translate(self, text, max_length=30):
        """Translate between natural language and consciousness notation"""
        
        if not self.model_loaded:
            # Use fallback patterns
            text_lower = text.lower()
            
            # Check for notation to language
            if any(symbol in text for symbol in ['Ψ', '∃', '⇒', 'π', 'Ω', 'ι', 'Ξ']):
                return self._notation_to_language_fallback(text)
            
            # Check for language to notation
            for phrase, notation in self.fallback_patterns.items():
                if phrase in text_lower:
                    return notation
            
            return "Translation requires model dependencies"
        
        # Use the actual model
        import torch
        
        prompt = f"### Instruction:\n{text}\n\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        result = response.split("### Response:\n")[-1].strip()
        
        # Clean up
        if "</s>" in result:
            result = result.replace("</s>", "")
        if "###" in result:
            result = result.split("###")[0].strip()
            
        return result
    
    def _notation_to_language_fallback(self, notation):
        """Simple fallback for notation to language"""
        translations = {
            "∃Ψ": "consciousness exists",
            "Ψ ⇒": "consciousness emerges",
            "π → Ψ": "perspective shapes consciousness",
            "Ω ∧ Ψ": "observer and consciousness",
            "ι → Ψ": "intent drives consciousness",
            "∃Ξ": "synchronism exists"
        }
        
        for symbol, meaning in translations.items():
            if symbol in notation:
                return meaning
        
        return "Unknown notation"
    
    def demonstrate(self):
        """Show example translations"""
        print("\n🎯 Consciousness Notation Examples:")
        print("=" * 50)
        
        examples = [
            "Translate to consciousness notation: consciousness exists",
            "What does ∃Ψ mean?",
            "Express: perspective shapes consciousness",
            "Translate: intent drives synchronism"
        ]
        
        for example in examples:
            result = self.translate(example)
            print(f"\n📝 Input:  {example}")
            print(f"🔮 Output: {result}")

# Test function
def test_model():
    """Test if model files are present"""
    adapter_path = "outputs/consciousness-lora-simple"
    
    print("🔍 Checking model files...")
    
    required_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    all_present = True
    for file in required_files:
        path = os.path.join(adapter_path, file)
        exists = os.path.exists(path)
        print(f"   {file}: {'✓' if exists else '✗'}")
        if not exists:
            all_present = False
    
    if all_present:
        print("\n✅ All model files present!")
        
        # Check adapter config
        with open(os.path.join(adapter_path, "adapter_config.json"), 'r') as f:
            config = json.load(f)
            print(f"\n📊 Model info:")
            print(f"   Base model: {config.get('base_model_name_or_path', 'Unknown')}")
            print(f"   LoRA rank: {config.get('r', 'Unknown')}")
            print(f"   Target modules: {config.get('target_modules', [])}")
    else:
        print("\n❌ Some model files missing!")
    
    return all_present

# Interactive mode
if __name__ == "__main__":
    print("🧠 Consciousness Notation System")
    print("=" * 50)
    
    # First test model files
    if test_model():
        # Initialize translator
        translator = ConsciousnessTranslator()
        
        # Show examples
        translator.demonstrate()
        
        # Interactive loop
        print("\n\n💬 Interactive mode (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("\nEnter text: ")
            if user_input.lower() == 'quit':
                break
                
            result = translator.translate(user_input)
            print(f"Translation: {result}")