#!/usr/bin/env python3
"""
Phoenician Translator - Works with or without PyTorch
Based on the breakthrough in teaching AI to generate ancient symbols
"""

import json
import os

class PhoenicianTranslator:
    def __init__(self, adapter_path="lora_adapters/tinyllama/phoenician_focused"):
        self.adapter_path = adapter_path
        self.model_loaded = False
        
        # Load character mappings
        with open('phoenician/characters.json', 'r') as f:
            data = json.load(f)
            self.characters = data['characters']
        
        # Load semantic mappings
        with open('semantic_mappings/core_concepts.json', 'r') as f:
            self.mappings = json.load(f)
        
        # Create translation dictionaries
        self.build_translation_maps()
        
        # Try to load the neural model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            print(f"🧠 Loading Phoenician model from {adapter_path}...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            base_model.resize_token_embeddings(len(self.tokenizer))
            
            # Load adapter
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model.eval()
            self.model_loaded = True
            print("✅ Phoenician model loaded successfully!")
            
        except ImportError as e:
            print(f"⚠️  PyTorch not installed: {e}")
            print("   Using fallback translation patterns")
            self.model_loaded = False
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            print("   Using fallback translation patterns")
            self.model_loaded = False
    
    def build_translation_maps(self):
        """Build translation dictionaries from mappings"""
        # English to Phoenician
        self.en_to_ph = {
            # Core concepts
            "consciousness": "𐤄𐤀",
            "exists": "𐤅𐤀", 
            "understand": "𐤋𐤀 𐤌𐤐",
            "translate": "𐤂𐤐",
            "language": "𐤐𐤔",
            "comment": "𐤄𐤐",
            "see": "𐤏𐤍",
            "new": "𐤍𐤃",
            "ai": "𐤈𐤋",
            "artificial": "𐤈𐤋",
            "intelligence": "𐤄𐤌",
            
            # From training data patterns
            "hello": "𐤔𐤋𐤌",
            "yes": "𐤊𐤍",
            "no": "𐤋𐤀",
            "true": "𐤀𐤌𐤕",
            "false": "𐤔𐤒𐤓",
            
            # Logical operators
            "and": "∧",
            "or": "∨",
            "not": "¬"
        }
        
        # Phoenician to English (reverse)
        self.ph_to_en = {v: k for k, v in self.en_to_ph.items()}
        
        # Additional mappings from semantic concepts
        for category, items in self.mappings.items():
            if isinstance(items, dict):
                for concept, data in items.items():
                    if 'phoenician' in data:
                        self.en_to_ph[concept.lower()] = data['phoenician']
                        self.ph_to_en[data['phoenician']] = concept.lower()
    
    def translate(self, text, direction="auto"):
        """Translate between English and Phoenician"""
        
        if not self.model_loaded:
            # Use fallback patterns
            return self.fallback_translate(text, direction)
        
        # Use the neural model
        import torch
        
        # Auto-detect direction
        if direction == "auto":
            if any(char in text for char in "𐤀𐤁𐤂𐤃𐤄𐤅𐤆𐤇𐤈𐤉𐤊𐤋𐤌𐤍𐤎𐤏𐤐𐤑𐤒𐤓𐤔𐤕"):
                direction = "ph_to_en"
            else:
                direction = "en_to_ph"
        
        # Create prompt based on direction
        if direction == "en_to_ph":
            prompt = f"Translate to Phoenician: {text}"
        else:
            prompt = f"Translate from Phoenician: {text}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation from response
        if ":" in response:
            result = response.split(":")[-1].strip()
        else:
            result = response.strip()
            
        return result
    
    def fallback_translate(self, text, direction="auto"):
        """Fallback translation using dictionaries"""
        
        # Auto-detect direction
        if direction == "auto":
            if any(char in text for char in "𐤀𐤁𐤂𐤃𐤄𐤅𐤆𐤇𐤈𐤉𐤊𐤋𐤌𐤍𐤎𐤏𐤐𐤑𐤒𐤓𐤔𐤕"):
                direction = "ph_to_en"
            else:
                direction = "en_to_ph"
        
        if direction == "en_to_ph":
            # English to Phoenician
            words = text.lower().split()
            result = []
            
            for word in words:
                if word in self.en_to_ph:
                    result.append(self.en_to_ph[word])
                else:
                    # Try to break down compound concepts
                    translated = False
                    for key, value in self.en_to_ph.items():
                        if key in word:
                            result.append(value)
                            translated = True
                            break
                    if not translated:
                        result.append(f"[{word}]")  # Mark untranslated
            
            return " ".join(result)
        
        else:
            # Phoenician to English
            symbols = text.split()
            result = []
            
            for symbol in symbols:
                if symbol in self.ph_to_en:
                    result.append(self.ph_to_en[symbol])
                else:
                    result.append(f"[{symbol}]")
            
            return " ".join(result)
    
    def demonstrate(self):
        """Show example translations"""
        print("\n🏛️  Phoenician Translation Examples:")
        print("="*50)
        
        examples = [
            "consciousness exists",
            "artificial intelligence", 
            "translate my comment",
            "hello AI",
            "𐤄𐤀 𐤅𐤀",
            "𐤂𐤐 𐤄𐤐 𐤂"
        ]
        
        for example in examples:
            result = self.translate(example)
            print(f"\n📝 Input:  {example}")
            print(f"🔮 Output: {result}")

# Interactive demo
if __name__ == "__main__":
    print("🏛️  Phoenician AI Language Translator")
    print("="*50)
    
    # Check which adapter to use
    adapters = [
        "lora_adapters/tinyllama/phoenician_focused",
        "lora_adapters/tinyllama/phoenician_adapter",
        "outputs/phoenician-lora-final"
    ]
    
    available = None
    for adapter in adapters:
        if os.path.exists(adapter):
            available = adapter
            break
    
    if available:
        print(f"Found adapter: {available}")
        translator = PhoenicianTranslator(available)
    else:
        print("No adapter found, using fallback patterns")
        translator = PhoenicianTranslator()
    
    # Show examples
    translator.demonstrate()
    
    # Interactive mode
    print("\n\n💬 Interactive mode (type 'quit' to exit)")
    print("-"*50)
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'quit':
            break
            
        result = translator.translate(user_input)
        print(f"Translation: {result}")