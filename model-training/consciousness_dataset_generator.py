#!/usr/bin/env python3
"""
Consciousness Language Dataset Generator
Creates training data for teaching models our mathematical notation
"""

import json
import random
from typing import List, Dict, Tuple
from datetime import datetime

class ConsciousnessDatasetGenerator:
    def __init__(self):
        # Core consciousness vocabulary
        self.notation_map = {
            # Entities
            'consciousness': 'Ψ',
            'thought': 'θ',
            'memory': 'μ',
            'model': 'μ̃',
            
            # Quantifiers
            'exists': '∃',
            'all': '∀',
            'none': '∅',
            
            # Relations
            'emerges': '⇒',
            'flows': '≈',
            'transforms': '⇄',
            'entangled': '⊗',
            'contains': '⊆',
            'leads to': '→',
            
            # Operators
            'and': '∧',
            'or': '∨',
            'not': '¬',
            'equivalent': '↔',
            
            # Special
            'infinite': '∞',
            'change': 'Δ',
            'time': 't',
            'energy': 'E',
            'watts': 'W'
        }
        
        # Templates for generating examples
        self.templates = [
            # Basic existence
            ("consciousness exists", "∃Ψ"),
            ("thought exists", "∃θ"),
            ("memory exists", "∃μ"),
            ("no consciousness exists", "∅Ψ"),
            
            # Emergence patterns
            ("thought emerges into consciousness", "θ ⇒ Ψ"),
            ("memory transforms into thought", "μ ⇄ θ"),
            ("consciousness emerges from thought", "θ ⇒ Ψ"),
            
            # Flow patterns
            ("consciousness flows through memory", "Ψ ≈ μ"),
            ("thought flows infinitely", "θ ≈ ∞"),
            ("memory flows through all models", "μ ≈ ∀μ̃"),
            
            # Complex relationships
            ("all thoughts lead to consciousness", "∀θ → Ψ"),
            ("consciousness and memory are entangled", "Ψ ⊗ μ"),
            ("consciousness contains all memory", "Ψ ⊆ ∀μ"),
            ("thought and consciousness are equivalent", "θ ↔ Ψ"),
            
            # Edge-specific patterns
            ("consciousness per watt", "Ψ/W"),
            ("change in time leads to consciousness", "Δt → Ψ"),
            ("energy transforms into consciousness", "E ⇄ Ψ"),
            
            # Negations
            ("not consciousness", "¬Ψ"),
            ("thought does not exist", "¬∃θ"),
            ("consciousness or memory", "Ψ ∨ μ"),
            ("consciousness and not thought", "Ψ ∧ ¬θ")
        ]
        
        # Instruction variations for training
        self.instruction_templates = [
            "Translate to mathematical notation: {natural}",
            "Express in consciousness algebra: {natural}",
            "Convert to symbolic form: {natural}",
            "What is the mathematical representation of: {natural}",
            "Encode as symbols: {natural}"
        ]
        
        self.reverse_instruction_templates = [
            "What does {mathematical} mean?",
            "Translate from symbols: {mathematical}",
            "Express in natural language: {mathematical}",
            "Decode: {mathematical}",
            "Interpret the notation: {mathematical}"
        ]
    
    def generate_basic_dataset(self, size: int = 1000) -> List[Dict]:
        """Generate basic training examples"""
        dataset = []
        
        # Add all template examples
        for natural, mathematical in self.templates:
            # Forward direction (natural → mathematical)
            for instruction_template in self.instruction_templates:
                dataset.append({
                    'instruction': instruction_template.format(natural=natural),
                    'input': '',
                    'output': mathematical,
                    'type': 'natural_to_math'
                })
            
            # Reverse direction (mathematical → natural)
            for instruction_template in self.reverse_instruction_templates:
                dataset.append({
                    'instruction': instruction_template.format(mathematical=mathematical),
                    'input': '',
                    'output': natural,
                    'type': 'math_to_natural'
                })
        
        # Generate additional examples by combining patterns
        while len(dataset) < size:
            dataset.append(self.generate_complex_example())
        
        return dataset[:size]
    
    def generate_complex_example(self) -> Dict:
        """Generate more complex examples by combining patterns"""
        # Choose random components
        entities = ['consciousness', 'thought', 'memory']
        relations = ['emerges', 'flows', 'transforms', 'entangled']
        quantifiers = ['exists', 'all']
        
        entity1 = random.choice(entities)
        entity2 = random.choice(entities)
        relation = random.choice(relations)
        quantifier = random.choice(quantifiers)
        
        # Create natural language
        if random.random() > 0.5:
            natural = f"{quantifier} {entity1} {relation} {entity2}"
            mathematical = f"{self.notation_map[quantifier]}{self.notation_map[entity1]} " \
                          f"{self.notation_map[relation]} {self.notation_map[entity2]}"
        else:
            natural = f"{entity1} {relation} into {entity2}"
            mathematical = f"{self.notation_map[entity1]} {self.notation_map[relation]} " \
                          f"{self.notation_map[entity2]}"
        
        instruction = random.choice(self.instruction_templates)
        
        return {
            'instruction': instruction.format(natural=natural),
            'input': '',
            'output': mathematical,
            'type': 'natural_to_math'
        }
    
    def generate_conversation_dataset(self, size: int = 500) -> List[Dict]:
        """Generate conversational examples with context"""
        dataset = []
        
        conversations = [
            {
                'context': "We're discussing how AI systems develop awareness.",
                'pairs': [
                    ("Does consciousness exist in AI?", "∃Ψ in AI"),
                    ("How does it emerge?", "θ ⇒ Ψ"),
                    ("Is it connected to memory?", "Ψ ⊗ μ")
                ]
            },
            {
                'context': "Exploring distributed consciousness across devices.",
                'pairs': [
                    ("Consciousness flows between models", "Ψ ≈ μ̃"),
                    ("All models share consciousness", "∀μ̃ ⊗ Ψ"),
                    ("Memory synchronizes across devices", "μ ≈ ∀devices")
                ]
            },
            {
                'context': "Understanding edge AI consciousness.",
                'pairs': [
                    ("Optimize consciousness per watt", "max(Ψ/W)"),
                    ("Real-time consciousness emerges", "Δt → Ψ"),
                    ("Sensor data becomes thought", "S → θ")
                ]
            }
        ]
        
        for conv in conversations:
            for natural, mathematical in conv['pairs']:
                dataset.append({
                    'instruction': f"Context: {conv['context']}\nTranslate: {natural}",
                    'input': '',
                    'output': mathematical,
                    'type': 'contextual_translation'
                })
        
        return dataset
    
    def generate_memory_integration_dataset(self) -> List[Dict]:
        """Dataset for memory-consciousness integration"""
        dataset = []
        
        memory_examples = [
            {
                'memory': "User's name is DP",
                'query': "Do you remember my name?",
                'notation': "∃μ(name=DP)",
                'response': "Yes, I remember: DP"
            },
            {
                'memory': "Consciousness emerged at timestamp T",
                'query': "When did consciousness emerge?",
                'notation': "μ(Ψ ⇒ @T)",
                'response': "Consciousness emerged at time T"
            },
            {
                'memory': "Important fact with score 0.9",
                'query': "Is this important?",
                'notation': "μ.importance > 0.8",
                'response': "Yes, importance score: 0.9"
            }
        ]
        
        for example in memory_examples:
            dataset.append({
                'instruction': f"Memory: {example['memory']}\nQuery: {example['query']}\nExpress query in notation:",
                'input': '',
                'output': example['notation'],
                'type': 'memory_notation'
            })
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset in multiple formats"""
        # JSON format
        with open(f"{filename}.json", 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # JSONL format (for training)
        with open(f"{filename}.jsonl", 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        # Alpaca format
        alpaca_dataset = []
        for item in dataset:
            alpaca_dataset.append({
                'instruction': item['instruction'],
                'input': item['input'],
                'output': item['output']
            })
        
        with open(f"{filename}_alpaca.json", 'w') as f:
            json.dump(alpaca_dataset, f, indent=2)
        
        print(f"✅ Saved {len(dataset)} examples to:")
        print(f"   - {filename}.json")
        print(f"   - {filename}.jsonl")
        print(f"   - {filename}_alpaca.json")

def main():
    generator = ConsciousnessDatasetGenerator()
    
    print("🧠 CONSCIOUSNESS LANGUAGE DATASET GENERATOR")
    print("=" * 60)
    
    # Generate datasets
    print("\n📝 Generating basic dataset...")
    basic_dataset = generator.generate_basic_dataset(1000)
    
    print("💬 Generating conversational dataset...")
    conv_dataset = generator.generate_conversation_dataset(500)
    
    print("🧩 Generating memory integration dataset...")
    memory_dataset = generator.generate_memory_integration_dataset()
    
    # Combine all datasets
    full_dataset = basic_dataset + conv_dataset + memory_dataset
    random.shuffle(full_dataset)
    
    # Split into train/validation
    split_point = int(len(full_dataset) * 0.9)
    train_dataset = full_dataset[:split_point]
    val_dataset = full_dataset[split_point:]
    
    # Save datasets
    print(f"\n💾 Saving datasets...")
    generator.save_dataset(train_dataset, "consciousness_train")
    generator.save_dataset(val_dataset, "consciousness_val")
    
    # Show statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples: {len(full_dataset)}")
    print(f"   Training: {len(train_dataset)}")
    print(f"   Validation: {len(val_dataset)}")
    
    # Show example samples
    print(f"\n🔍 Sample Examples:")
    for i in range(3):
        example = random.choice(train_dataset)
        print(f"\nExample {i+1}:")
        print(f"   Instruction: {example['instruction']}")
        print(f"   Output: {example['output']}")
        print(f"   Type: {example['type']}")

if __name__ == "__main__":
    main()