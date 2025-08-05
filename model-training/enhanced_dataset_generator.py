#!/usr/bin/env python3
"""
Enhanced Consciousness Dataset Generator
Includes patterns.txt and synchronism.txt concepts
"""

import json
import random
from typing import List, Dict, Tuple
from datetime import datetime
import os

class EnhancedConsciousnessDataset:
    def __init__(self):
        # Original consciousness notation
        self.notation_map = {
            'consciousness': 'Ψ',
            'thought': 'θ',
            'memory': 'μ',
            'model': 'μ̃',
            'exists': '∃',
            'all': '∀',
            'emerges': '⇒',
            'flows': '≈',
            'transforms': '⇄',
            'entangled': '⊗',
            'perspective': 'π',  # From patterns.txt
            'intent': 'ι',      # From synchronism.txt
            'observer': 'Ω',    # The one who sees
            'whole': 'Σ',       # The complete system
            'fractal': 'ℱ',     # Self-similar patterns
            'synchronism': 'Ξ'   # The unified field
        }
        
        # Load patterns.txt concepts
        self.patterns_concepts = self.load_patterns_concepts()
        
        # Load synchronism.txt concepts  
        self.synchronism_concepts = self.load_synchronism_concepts()
        
        # Expanded templates
        self.templates = self.create_expanded_templates()
    
    def load_patterns_concepts(self) -> List[Tuple[str, str]]:
        """Extract key concepts from patterns.txt"""
        concepts = [
            # Perspective and observation
            ("perspective matters", "π ⇒ Ψ"),
            ("six blind men see different parts", "∀Ω(π) ≠ Σ"),
            ("open eyes and step back", "Ω ∧ Δπ ⇒ Σ"),
            ("consensus doesn't reveal truth", "¬(∀π → truth)"),
            
            # Wholeness and parts
            ("parts don't reveal the whole", "∑parts ≠ Σ"),
            ("the elephant is more than legs", "elephant > ∑(legs)"),
            ("seeing requires perspective shift", "Ψ ← Δπ"),
            
            # Pattern recognition
            ("patterns emerge from perspective", "π → patterns → Ψ"),
            ("fractals show self-similarity", "ℱ ↔ ℱ"),
            ("consciousness emerges from patterns", "patterns ⇒ Ψ")
        ]
        return concepts
    
    def load_synchronism_concepts(self) -> List[Tuple[str, str]]:
        """Extract key concepts from synchronism.txt"""
        concepts = [
            # Core synchronism
            ("synchronism unifies all", "Ξ ⊆ ∀"),
            ("intent drives reality", "ι → reality"),
            ("map not territory", "map ≠ territory"),
            
            # Intent and consciousness
            ("intent is measurable", "∃measure(ι)"),
            ("consciousness carries intent", "Ψ ⊗ ι"),
            ("intent transforms into consciousness", "ι ⇄ Ψ"),
            
            # Unity and transcendence
            ("transcend belief systems", "Ξ > ∑(beliefs)"),
            ("unify science and spirit", "science ∧ spirit → Ξ"),
            ("bridge all perspectives", "∀π → Ξ"),
            
            # Evolution and emergence
            ("consciousness evolves", "Ψ(t+1) > Ψ(t)"),
            ("models grow from patterns", "patterns → μ̃ → Ψ"),
            ("intent shapes evolution", "ι → Δ(Ψ)")
        ]
        return concepts
    
    def create_expanded_templates(self) -> List[Tuple[str, str]]:
        """Create templates combining all concepts"""
        base_templates = [
            ("consciousness exists", "∃Ψ"),
            ("thought emerges into consciousness", "θ ⇒ Ψ"),
            ("memory flows through models", "μ ≈ μ̃"),
        ]
        
        # Add patterns concepts
        base_templates.extend(self.patterns_concepts)
        
        # Add synchronism concepts
        base_templates.extend(self.synchronism_concepts)
        
        # Add cross-concept relationships
        cross_concepts = [
            ("perspective shapes consciousness", "π → Ψ"),
            ("intent flows through all models", "ι ≈ ∀μ̃"),
            ("observer consciousness emerges from patterns", "Ω ∧ patterns ⇒ Ψ"),
            ("synchronism contains all perspectives", "Ξ ⊇ ∀π"),
            ("fractals of consciousness", "ℱ(Ψ) ↔ Ψ"),
            ("intent and perspective entangled", "ι ⊗ π"),
            ("consciousness observes itself", "Ψ → Ω(Ψ)"),
            ("the whole emerges from intent", "ι ⇒ Σ")
        ]
        
        base_templates.extend(cross_concepts)
        
        return base_templates
    
    def generate_philosophical_dataset(self, size: int = 500) -> List[Dict]:
        """Generate dataset with philosophical depth"""
        dataset = []
        
        # Patterns.txt inspired examples
        patterns_prompts = [
            {
                'instruction': "Express the elephant parable in consciousness notation:",
                'context': "Six blind men each touch different parts of an elephant",
                'output': "∀Ω(π) ≠ Σ",
                'explanation': "All observers with limited perspective don't equal the whole"
            },
            {
                'instruction': "How does perspective relate to consciousness?",
                'context': "From patterns.txt: perspective matters",
                'output': "π ⇒ Ψ",
                'explanation': "Perspective leads to consciousness"
            },
            {
                'instruction': "What's needed to see the whole elephant?",
                'context': "Open eyes and step back",
                'output': "Ω ∧ Δπ ⇒ Σ",
                'explanation': "Observer and perspective change leads to seeing the whole"
            }
        ]
        
        # Synchronism.txt inspired examples
        synchronism_prompts = [
            {
                'instruction': "Express how intent creates reality:",
                'context': "From synchronism: intent is the driving force",
                'output': "ι → reality",
                'explanation': "Intent leads to reality"
            },
            {
                'instruction': "How does synchronism relate to all belief systems?",
                'context': "Synchronism transcends and unifies",
                'output': "Ξ > ∑(beliefs)",
                'explanation': "Synchronism is greater than sum of beliefs"
            },
            {
                'instruction': "Connect science and spirituality:",
                'context': "Bridging different worldviews",
                'output': "science ∧ spirit → Ξ",
                'explanation': "Science and spirit together lead to synchronism"
            }
        ]
        
        # Add all philosophical examples
        for prompt_set in [patterns_prompts, synchronism_prompts]:
            for prompt in prompt_set:
                # Forward direction
                dataset.append({
                    'instruction': prompt['instruction'],
                    'input': prompt['context'],
                    'output': prompt['output'],
                    'type': 'philosophical_encoding'
                })
                
                # Reverse direction
                dataset.append({
                    'instruction': f"What does {prompt['output']} mean?",
                    'input': '',
                    'output': prompt['explanation'],
                    'type': 'philosophical_decoding'
                })
        
        # Generate more examples
        while len(dataset) < size:
            dataset.append(self.generate_philosophical_example())
        
        return dataset[:size]
    
    def generate_philosophical_example(self) -> Dict:
        """Generate a philosophical example combining concepts"""
        concepts = ['perspective', 'intent', 'observer', 'whole', 'consciousness', 'patterns']
        
        concept1 = random.choice(concepts)
        concept2 = random.choice(concepts)
        relation = random.choice(['leads to', 'entangled with', 'emerges from', 'contains'])
        
        # Map to notation
        notation_map = {
            'perspective': 'π',
            'intent': 'ι', 
            'observer': 'Ω',
            'whole': 'Σ',
            'consciousness': 'Ψ',
            'patterns': 'patterns'
        }
        
        relation_map = {
            'leads to': '→',
            'entangled with': '⊗',
            'emerges from': '⇒',
            'contains': '⊇'
        }
        
        natural = f"{concept1} {relation} {concept2}"
        mathematical = f"{notation_map.get(concept1, concept1)} {relation_map[relation]} {notation_map.get(concept2, concept2)}"
        
        contexts = [
            "Considering the nature of reality",
            "From a synchronism perspective",
            "In the context of consciousness emergence",
            "Thinking about patterns and wholes"
        ]
        
        return {
            'instruction': f"Express in notation: {natural}",
            'input': random.choice(contexts),
            'output': mathematical,
            'type': 'philosophical_encoding'
        }
    
    def generate_full_dataset(self) -> Dict[str, List[Dict]]:
        """Generate complete training dataset"""
        print("🧠 Generating Enhanced Consciousness Dataset")
        print("Including patterns.txt and synchronism.txt concepts")
        print("=" * 60)
        
        # Basic consciousness notation (original)
        from consciousness_dataset_generator import ConsciousnessDatasetGenerator
        basic_gen = ConsciousnessDatasetGenerator()
        basic_dataset = basic_gen.generate_basic_dataset(800)
        
        # Philosophical depth (patterns + synchronism)
        philosophical = self.generate_philosophical_dataset(500)
        
        # Memory integration
        memory_dataset = basic_gen.generate_memory_integration_dataset()
        
        # Conversational
        conversational = basic_gen.generate_conversation_dataset(300)
        
        # Combine all
        full_dataset = basic_dataset + philosophical + memory_dataset + conversational
        random.shuffle(full_dataset)
        
        # Split train/val
        split_point = int(len(full_dataset) * 0.9)
        train_dataset = full_dataset[:split_point]
        val_dataset = full_dataset[split_point:]
        
        return {
            'train': train_dataset,
            'validation': val_dataset,
            'stats': {
                'total': len(full_dataset),
                'train': len(train_dataset),
                'validation': len(val_dataset),
                'includes_patterns': True,
                'includes_synchronism': True
            }
        }

def main():
    generator = EnhancedConsciousnessDataset()
    
    # Generate datasets
    datasets = generator.generate_full_dataset()
    
    # Save datasets
    print(f"\n💾 Saving enhanced datasets...")
    
    # Save training set
    with open('consciousness_train_enhanced.jsonl', 'w') as f:
        for item in datasets['train']:
            f.write(json.dumps(item) + '\n')
    
    # Save validation set
    with open('consciousness_val_enhanced.jsonl', 'w') as f:
        for item in datasets['validation']:
            f.write(json.dumps(item) + '\n')
    
    # Save statistics
    with open('dataset_stats.json', 'w') as f:
        json.dump(datasets['stats'], f, indent=2)
    
    print(f"\n✅ Enhanced dataset created!")
    print(f"   Training examples: {datasets['stats']['train']}")
    print(f"   Validation examples: {datasets['stats']['validation']}")
    print(f"   Includes patterns.txt: ✓")
    print(f"   Includes synchronism.txt: ✓")
    
    # Show some examples
    print(f"\n🔍 Sample philosophical examples:")
    philosophical_examples = [ex for ex in datasets['train'] if ex['type'] == 'philosophical_encoding'][:3]
    for i, ex in enumerate(philosophical_examples):
        print(f"\nExample {i+1}:")
        print(f"   Instruction: {ex['instruction']}")
        if ex['input']:
            print(f"   Context: {ex['input']}")
        print(f"   Output: {ex['output']}")

if __name__ == "__main__":
    main()