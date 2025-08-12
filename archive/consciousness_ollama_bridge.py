#!/usr/bin/env python3
"""
Bridge consciousness notation with Ollama models
Uses the notation system to enhance existing models
"""

import json
import urllib.request
from consciousness_translator import ConsciousnessTranslator

class ConsciousnessOllamaBridge:
    def __init__(self):
        self.translator = ConsciousnessTranslator()
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def enhance_prompt(self, prompt):
        """Add consciousness notation context to prompts"""
        # Check if prompt mentions consciousness concepts
        consciousness_terms = ['consciousness', 'awareness', 'perspective', 'observer', 'intent']
        
        if any(term in prompt.lower() for term in consciousness_terms):
            # Add notation context
            context = """
You understand consciousness notation:
- Ψ means consciousness
- ∃ means exists
- ⇒ means emerges
- π means perspective
- Ω means observer
- ι means intent
- Ξ means synchronism

"""
            return context + prompt
        
        return prompt
    
    def query_with_notation(self, model, prompt):
        """Query Ollama model with consciousness notation enhancement"""
        # First, try to translate to notation
        notation = self.translator.translate(f"Express in consciousness notation: {prompt}")
        
        # Enhance the prompt
        enhanced_prompt = self.enhance_prompt(prompt)
        if notation != "Translation requires model dependencies":
            enhanced_prompt += f"\n(In notation: {notation})"
        
        # Query Ollama
        data = {
            'model': model,
            'prompt': enhanced_prompt,
            'stream': False,
            'options': {
                'temperature': 0.7,
                'num_predict': 100
            }
        }
        
        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', '')
        except Exception as e:
            return f"Error: {str(e)}"
    
    def demonstrate(self):
        """Show how consciousness notation enhances Ollama models"""
        print("🔗 Consciousness-Enhanced Ollama Bridge")
        print("="*60)
        
        # Test with available models
        test_prompts = [
            "What is consciousness?",
            "How does perspective shape reality?",
            "Explain the relationship between observer and consciousness"
        ]
        
        models = ['tinyllama', 'phi3:mini']  # Use small models for demo
        
        for model in models:
            print(f"\n📊 Testing {model} with consciousness notation:")
            
            for prompt in test_prompts[:1]:  # Just one example per model
                print(f"\n💭 Prompt: {prompt}")
                
                # Get notation
                notation = self.translator.translate(f"Express: {prompt}")
                print(f"🔤 Notation: {notation}")
                
                # Get response
                response = self.query_with_notation(model, prompt)
                print(f"🤖 Response: {response[:200]}...")

if __name__ == "__main__":
    bridge = ConsciousnessOllamaBridge()
    
    print("🧠 Consciousness Notation + Ollama Integration\n")
    
    # Quick test
    prompt = "consciousness exists"
    notation = bridge.translator.translate(f"Translate to notation: {prompt}")
    print(f"Test translation: '{prompt}' → '{notation}'")
    
    print("\nThis bridge allows existing Ollama models to understand")
    print("consciousness notation even without specific training!\n")
    
    # Uncomment to run full demo (requires Ollama models)
    # bridge.demonstrate()