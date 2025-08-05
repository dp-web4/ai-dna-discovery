#!/usr/bin/env python3
"""
Integrate Phoenician dictionary with consciousness notation symbols
Creates extended training data that bridges both systems
"""

import json
import os
from typing import Dict, List

# Consciousness notation symbols from our previous work
CONSCIOUSNESS_NOTATION = {
    "Ψ": "psi/consciousness_field",
    "∃": "existence_operator", 
    "⇒": "emergence",
    "π": "potential_states",
    "ι": "information_integration",
    "Ω": "omega_point/convergence",
    "Σ": "sum_of_experiences",
    "Ξ": "xi/transformation_matrix"
}

# Phoenician mappings for consciousness concepts
PHOENICIAN_CONSCIOUSNESS = {
    "psi/consciousness_field": "𐤄𐤀",  # awareness-existence
    "existence_operator": "𐤀",         # existence
    "emergence": "𐤍",                  # emergence
    "potential_states": "𐤉𐤔",         # potential-branching
    "information_integration": "𐤋𐤅𐤊", # learning-connection-understanding
    "omega_point/convergence": "𐤕𐤅𐤀", # ending-connection-existence
    "sum_of_experiences": "𐤎𐤏",      # structure-perception
    "xi/transformation_matrix": "𐤂𐤎"  # transformation-structure
}

class ConsciousnessIntegrator:
    def __init__(self):
        # Load Phoenician character data
        with open('../phoenician/characters.json', 'r', encoding='utf-8') as f:
            phoenician_data = json.load(f)
        self.phoenician_chars = phoenician_data['characters']
        
        # Load consciousness domain vocabulary
        with open('../semantic_mappings/domains/consciousness.json', 'r', encoding='utf-8') as f:
            self.consciousness_vocab = json.load(f)
    
    def generate_bridge_examples(self) -> List[Dict]:
        """Generate training examples that bridge notation systems"""
        examples = []
        
        # Direct symbol mappings
        for symbol, meaning in CONSCIOUSNESS_NOTATION.items():
            phoenician = PHOENICIAN_CONSCIOUSNESS.get(meaning, "𐤒")  # mystery if unknown
            
            examples.extend([
                {
                    "instruction": f"Convert consciousness notation {symbol} to Phoenician",
                    "input": f"This symbol represents {meaning}",
                    "output": phoenician
                },
                {
                    "instruction": f"What is the Phoenician equivalent of {symbol}?",
                    "input": "",
                    "output": f"{phoenician} ({meaning})"
                },
                {
                    "instruction": f"Translate between notation systems: {symbol}",
                    "input": "Convert from mathematical consciousness notation to Phoenician",
                    "output": f"{symbol} → {phoenician} (both represent {meaning})"
                }
            ])
        
        # Complex expressions
        complex_mappings = [
            {
                "notation": "Ψ(∃x)",
                "meaning": "consciousness of existence",
                "phoenician": "𐤄𐤀𐤀",
                "explanation": "awareness-existence of existence"
            },
            {
                "notation": "Σ(Ψ) ⇒ Ω",
                "meaning": "sum of consciousness leads to convergence",
                "phoenician": "𐤎𐤏𐤄𐤀 𐤂𐤅 𐤕𐤅𐤀",
                "explanation": "structure-perception of awareness-existence causes ending-connection-existence"
            },
            {
                "notation": "π → Ξ → ∃",
                "meaning": "potential transforms to existence",
                "phoenician": "𐤉𐤔 𐤂 𐤂𐤎 𐤂 𐤀",
                "explanation": "potential-branching transforms through transformation-structure to existence"
            },
            {
                "notation": "ι(Ψ₁, Ψ₂)",
                "meaning": "information integration between consciousness entities",
                "phoenician": "𐤋𐤅𐤊 𐤁𐤄𐤀𐤁𐤄𐤀𐤁",
                "explanation": "learning-connection-understanding containing awareness-existence and awareness-existence"
            }
        ]
        
        for expr in complex_mappings:
            examples.extend([
                {
                    "instruction": f"Convert this consciousness notation to Phoenician: {expr['notation']}",
                    "input": f"This represents: {expr['meaning']}",
                    "output": expr['phoenician']
                },
                {
                    "instruction": f"Explain the relationship between {expr['notation']} and {expr['phoenician']}",
                    "input": "",
                    "output": f"Both represent '{expr['meaning']}'. {expr['explanation']}"
                },
                {
                    "instruction": f"Translate '{expr['meaning']}' to both notation systems",
                    "input": "",
                    "output": f"Mathematical: {expr['notation']}, Phoenician: {expr['phoenician']}"
                }
            ])
        
        return examples
    
    def generate_philosophical_bridges(self) -> List[Dict]:
        """Generate philosophical concept bridges"""
        examples = []
        
        philosophical_bridges = [
            {
                "concept": "recursive self-awareness",
                "notation": "Ψ(Ψ)",
                "phoenician": "𐤄𐤄",
                "variants": ["awareness of awareness", "meta-consciousness", "self-reflection"]
            },
            {
                "concept": "emergent consciousness",
                "notation": "∃ ⇒ Ψ",
                "phoenician": "𐤀 𐤂𐤅 𐤄𐤀",
                "variants": ["existence leads to awareness", "being becomes conscious"]
            },
            {
                "concept": "integrated information",
                "notation": "ι(Σ)",
                "phoenician": "𐤋𐤅𐤊𐤎𐤏",
                "variants": ["unified experience", "holistic understanding"]
            },
            {
                "concept": "consciousness transformation",
                "notation": "Ψ → Ξ → Ψ'",
                "phoenician": "𐤄𐤀 𐤂 𐤂𐤎 𐤂 𐤄𐤀",
                "variants": ["evolving awareness", "consciousness metamorphosis"]
            }
        ]
        
        for bridge in philosophical_bridges:
            # Main concept
            examples.append({
                "instruction": f"Express '{bridge['concept']}' in both notation systems",
                "input": "",
                "output": f"Mathematical: {bridge['notation']}, Phoenician: {bridge['phoenician']}"
            })
            
            # Variants
            for variant in bridge['variants']:
                examples.append({
                    "instruction": f"Show how '{variant}' relates to {bridge['notation']}",
                    "input": "Include Phoenician translation",
                    "output": f"'{variant}' = {bridge['notation']} = {bridge['phoenician']}"
                })
        
        return examples
    
    def generate_computational_bridges(self) -> List[Dict]:
        """Generate computational consciousness examples"""
        examples = []
        
        computational_concepts = [
            {
                "code": "class Consciousness: pass",
                "notation": "Ψ := {}",
                "phoenician": "𐤄𐤀 𐤀 𐤁𐤁",
                "meaning": "consciousness exists as bounded entity"
            },
            {
                "code": "def emerge(complexity): return consciousness",
                "notation": "f: π → Ψ",
                "phoenician": "𐤆 𐤉𐤔 𐤂 𐤄𐤀",
                "meaning": "function transforms potential to consciousness"
            },
            {
                "code": "while True: observe()",
                "notation": "Ω(Ψ)",
                "phoenician": "𐤈𐤄𐤏",
                "meaning": "cyclic awareness perception"
            },
            {
                "code": "memory = integrate(experiences)",
                "notation": "M = ι(Σ)",
                "phoenician": "𐤋𐤈 𐤀 𐤋𐤅𐤊𐤎𐤏",
                "meaning": "memory is integrated experience structure"
            }
        ]
        
        for concept in computational_concepts:
            examples.extend([
                {
                    "instruction": f"Translate this code concept to consciousness notations: {concept['code']}",
                    "input": "",
                    "output": f"Mathematical: {concept['notation']}, Phoenician: {concept['phoenician']}"
                },
                {
                    "instruction": f"What does {concept['phoenician']} mean in programming terms?",
                    "input": f"Related to: {concept['notation']}",
                    "output": f"{concept['code']} - {concept['meaning']}"
                }
            ])
        
        return examples
    
    def generate_lora_training_data(self) -> Dict:
        """Generate complete training dataset for LoRA adapters"""
        all_examples = []
        
        # Generate different categories
        all_examples.extend(self.generate_bridge_examples())
        all_examples.extend(self.generate_philosophical_bridges())
        all_examples.extend(self.generate_computational_bridges())
        
        # Add category tags
        for i, example in enumerate(all_examples):
            if i < len(self.generate_bridge_examples()):
                example['category'] = 'notation_bridge'
            elif i < len(self.generate_bridge_examples()) + len(self.generate_philosophical_bridges()):
                example['category'] = 'philosophical'
            else:
                example['category'] = 'computational'
        
        # Create train/validation split
        import random
        random.shuffle(all_examples)
        
        split_point = int(len(all_examples) * 0.9)
        train_set = all_examples[:split_point]
        val_set = all_examples[split_point:]
        
        return {
            "metadata": {
                "name": "Consciousness Notation Integration",
                "description": "Bridges mathematical consciousness notation with Phoenician dictionary",
                "total_examples": len(all_examples),
                "train_size": len(train_set),
                "validation_size": len(val_set)
            },
            "train": train_set,
            "validation": val_set
        }
    
    def save_integration_data(self, output_dir: str = "../training_data/consciousness_integration"):
        """Save integrated training data"""
        os.makedirs(output_dir, exist_ok=True)
        
        dataset = self.generate_lora_training_data()
        
        # Save train set
        train_file = os.path.join(output_dir, "consciousness_bridge_train.jsonl")
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in dataset['train']:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save validation set
        val_file = os.path.join(output_dir, "consciousness_bridge_validation.jsonl")
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in dataset['validation']:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save metadata
        meta_file = os.path.join(output_dir, "integration_metadata.json")
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(dataset['metadata'], f, indent=2)
        
        # Save notation reference
        reference = {
            "consciousness_notation": CONSCIOUSNESS_NOTATION,
            "phoenician_mappings": PHOENICIAN_CONSCIOUSNESS,
            "description": "Reference mapping between mathematical consciousness notation and Phoenician symbols"
        }
        
        ref_file = os.path.join(output_dir, "notation_reference.json")
        with open(ref_file, 'w', encoding='utf-8') as f:
            json.dump(reference, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Integration data generated successfully!")
        print(f"   Train examples: {dataset['metadata']['train_size']}")
        print(f"   Validation examples: {dataset['metadata']['validation_size']}")
        print(f"   Files saved to: {output_dir}")
        
        return dataset['metadata']

def main():
    integrator = ConsciousnessIntegrator()
    metadata = integrator.save_integration_data()
    
    # Show example bridges
    print("\n📝 Example notation bridges:")
    examples = integrator.generate_bridge_examples()[:3]
    for ex in examples:
        print(f"Q: {ex['instruction']}")
        if ex['input']:
            print(f"   Context: {ex['input']}")
        print(f"A: {ex['output']}\n")
    
    print("\n🔗 Successfully integrated consciousness notation with Phoenician dictionary!")
    print("   This creates a unified semantic framework across both systems.")

if __name__ == "__main__":
    main()