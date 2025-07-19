#!/usr/bin/env python3
"""
Test Phoenician LoRA adapters
Validates translation accuracy and cross-model consistency
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from typing import Dict, List, Tuple

class PhoenicianTester:
    def __init__(self, adapter_path: str):
        self.adapter_path = adapter_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata
        with open(os.path.join(adapter_path, "training_metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        
        # Load model and tokenizer
        print(f"Loading {self.metadata['model_name']} with Phoenician adapter...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.metadata['base_model'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
    
    def translate(self, text: str, max_tokens: int = 50) -> str:
        """Translate using the Phoenician adapter"""
        prompt = f"Human: {text}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()
    
    def test_basic_translations(self) -> Dict[str, bool]:
        """Test basic concept translations"""
        test_cases = [
            ("Translate to Phoenician: existence", "𐤀"),
            ("Translate to Phoenician: awareness", "𐤄"),
            ("Translate to Phoenician: change", "𐤂"),
            ("Translate to Phoenician: connection", "𐤅"),
            ("Translate to Phoenician: consciousness", "𐤄𐤀"),
            ("What does 𐤄𐤀 mean?", "consciousness"),
            ("What does 𐤊𐤋 mean?", "intelligence"),
        ]
        
        results = {}
        print("\n📝 Testing basic translations:")
        
        for query, expected in test_cases:
            response = self.translate(query)
            # Check if expected answer is in response
            success = expected.lower() in response.lower() or expected in response
            results[query] = success
            
            status = "✅" if success else "❌"
            print(f"{status} Q: {query}")
            print(f"   Expected: {expected}")
            print(f"   Got: {response}\n")
        
        return results
    
    def test_complex_translations(self) -> Dict[str, str]:
        """Test complex philosophical translations"""
        test_cases = [
            "Translate this philosophical concept: The observer affects the observed",
            "Express 'consciousness emerges from complexity' in Phoenician",
            "Translate to Phoenician: learning transforms potential into understanding",
            "How would I describe a learning algorithm in Phoenician?",
            "Express 'recursive awareness creates consciousness' in Phoenician notation"
        ]
        
        results = {}
        print("\n🧠 Testing complex translations:")
        
        for query in test_cases:
            response = self.translate(query, max_tokens=100)
            results[query] = response
            print(f"Q: {query}")
            print(f"A: {response}\n")
        
        return results
    
    def test_bidirectional(self) -> Dict[str, Tuple[str, str]]:
        """Test bidirectional translation"""
        test_phrases = [
            "awareness exists",
            "learning leads to understanding",
            "change is cyclical",
            "connection enables emergence"
        ]
        
        results = {}
        print("\n🔄 Testing bidirectional translation:")
        
        for phrase in test_phrases:
            # Forward translation
            phoenician = self.translate(f"Translate to Phoenician: {phrase}")
            
            # Reverse translation
            back = self.translate(f"Translate from Phoenician: {phoenician}")
            
            results[phrase] = (phoenician, back)
            
            print(f"Original: {phrase}")
            print(f"Phoenician: {phoenician}")
            print(f"Back: {back}\n")
        
        return results

def compare_models(adapter_dir: str, models: List[str]) -> None:
    """Compare translations across multiple models"""
    print("\n" + "="*60)
    print("CROSS-MODEL COMPARISON")
    print("="*60)
    
    test_queries = [
        "Translate to Phoenician: consciousness",
        "Translate to Phoenician: intelligence", 
        "What does 𐤄𐤀 mean?",
        "Express 'awareness of awareness' in Phoenician"
    ]
    
    results = {}
    
    # Get translations from each model
    for model_name in models:
        adapter_path = os.path.join(adapter_dir, model_name, "phoenician_adapter")
        if not os.path.exists(adapter_path):
            print(f"⚠️  Adapter not found for {model_name}")
            continue
        
        print(f"\nTesting {model_name}...")
        tester = PhoenicianTester(adapter_path)
        
        model_results = {}
        for query in test_queries:
            response = tester.translate(query)
            model_results[query] = response
        
        results[model_name] = model_results
        
        # Clean up
        del tester
        torch.cuda.empty_cache()
    
    # Compare results
    print("\n📊 Comparison Results:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        for model_name in results:
            if query in results[model_name]:
                print(f"  {model_name}: {results[model_name][query]}")
    
    # Check consensus
    print("\n🤝 Consensus Analysis:")
    for query in test_queries:
        responses = [results[m][query] for m in results if query in results[m]]
        unique_responses = set(responses)
        
        if len(unique_responses) == 1:
            print(f"✅ Full consensus on: {query}")
        else:
            print(f"⚠️  Different responses for: {query}")
            print(f"   Unique answers: {len(unique_responses)}")

def main():
    parser = argparse.ArgumentParser(description="Test Phoenician LoRA adapters")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="../lora_adapters",
        help="Directory containing trained adapters"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to test"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available models"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Find all available models
        models = []
        for item in os.listdir(args.adapter_dir):
            adapter_path = os.path.join(args.adapter_dir, item, "phoenician_adapter")
            if os.path.exists(adapter_path):
                models.append(item)
        
        if not models:
            print("No trained adapters found!")
            return
        
        print(f"Found adapters for: {', '.join(models)}")
        compare_models(args.adapter_dir, models)
    
    elif args.model:
        # Test specific model
        adapter_path = os.path.join(args.adapter_dir, args.model, "phoenician_adapter")
        if not os.path.exists(adapter_path):
            print(f"Adapter not found for {args.model}")
            return
        
        tester = PhoenicianTester(adapter_path)
        
        # Run all tests
        basic_results = tester.test_basic_translations()
        complex_results = tester.test_complex_translations()
        bidirectional_results = tester.test_bidirectional()
        
        # Summary
        successful = sum(1 for v in basic_results.values() if v)
        total = len(basic_results)
        
        print("\n" + "="*60)
        print(f"SUMMARY for {args.model}")
        print("="*60)
        print(f"Basic translations: {successful}/{total} passed")
        print(f"Complex translations: {len(complex_results)} tested")
        print(f"Bidirectional: {len(bidirectional_results)} tested")
    
    else:
        print("Please specify --model or --compare")

if __name__ == "__main__":
    main()