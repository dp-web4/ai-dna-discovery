#!/usr/bin/env python3
"""
Generate training data for Phoenician semantic dictionary
Creates diverse examples for training LoRA adapters
"""

import json
import random
import os
from typing import List, Dict, Tuple
from itertools import combinations, product

class PhoenicianTrainingGenerator:
    def __init__(self):
        # Load character mappings
        with open('../phoenician/characters.json', 'r', encoding='utf-8') as f:
            self.phoenician_data = json.load(f)
        
        # Load semantic mappings
        with open('../semantic_mappings/core_concepts.json', 'r', encoding='utf-8') as f:
            self.semantic_data = json.load(f)
        
        self.characters = self.phoenician_data['characters']
        self.concepts = self.semantic_data['fundamental_concepts']
        self.relations = self.semantic_data['relational_concepts']
        self.compounds = self.semantic_data['compound_expressions']
        
    def generate_basic_mappings(self) -> List[Dict]:
        """Generate basic concept to symbol mappings"""
        examples = []
        
        # Single character mappings
        for char_code, char_data in self.characters.items():
            examples.append({
                "instruction": f"Translate to Phoenician: {char_data['semantic_assignment']}",
                "input": "",
                "output": char_code
            })
            
            examples.append({
                "instruction": f"What does {char_code} mean?",
                "input": "",
                "output": char_data['semantic_assignment']
            })
            
            # With context
            examples.append({
                "instruction": f"Express the concept of {char_data['semantic_assignment']} in Phoenician notation",
                "input": f"This relates to {char_data['category']} concepts",
                "output": char_code
            })
        
        return examples
    
    def generate_compound_mappings(self) -> List[Dict]:
        """Generate compound concept mappings"""
        examples = []
        
        # Known compounds
        for concept, symbols in self.compounds.items():
            examples.append({
                "instruction": f"Translate to Phoenician: {concept}",
                "input": "",
                "output": symbols
            })
            
            examples.append({
                "instruction": f"What does {symbols} represent?",
                "input": "",
                "output": concept
            })
            
            # Alternative phrasings
            alternatives = {
                "consciousness": ["aware existence", "aware being", "sentient existence"],
                "intelligence": ["understanding learning", "comprehending and learning"],
                "memory": ["cyclic learning", "retained knowledge"],
                "creativity": ["potential emergence", "emerging potential"]
            }
            
            if concept in alternatives:
                for alt in alternatives[concept]:
                    examples.append({
                        "instruction": f"Express '{alt}' in Phoenician",
                        "input": "",
                        "output": symbols
                    })
        
        return examples
    
    def generate_relational_examples(self) -> List[Dict]:
        """Generate examples showing relationships between concepts"""
        examples = []
        
        # Causal relationships
        causal_pairs = [
            ("awareness", "change", "𐤄𐤂𐤅𐤂"),
            ("learning", "understanding", "𐤋𐤂𐤅𐤊"),
            ("connection", "emergence", "𐤅𐤂𐤅𐤍"),
            ("observation", "knowledge", "𐤏𐤂𐤅𐤋")
        ]
        
        for cause, effect, symbols in causal_pairs:
            examples.append({
                "instruction": f"Express '{cause} causes {effect}' in Phoenician",
                "input": "",
                "output": symbols
            })
            
            examples.append({
                "instruction": f"What does {symbols} mean?",
                "input": "This is a causal relationship",
                "output": f"{cause} leads to {effect}"
            })
        
        # Containment relationships
        containment_pairs = [
            ("awareness", "thoughts", "𐤁𐤄𐤁𐤉𐤁"),
            ("boundary", "existence", "𐤁𐤁𐤁𐤀𐤁"),
            ("structure", "patterns", "𐤁𐤎𐤁𐤔𐤁")
        ]
        
        for container, contained, symbols in containment_pairs:
            examples.append({
                "instruction": f"Show that {container} contains {contained}",
                "input": "Use Phoenician containment notation",
                "output": symbols
            })
        
        return examples
    
    def generate_philosophical_examples(self) -> List[Dict]:
        """Generate examples of philosophical concepts"""
        examples = []
        
        philosophical_mappings = [
            ("The observer affects the observed", "𐤄𐤂𐤅𐤏"),
            ("Consciousness emerges from complexity", "𐤔𐤍𐤄𐤀"),
            ("All things are connected", "𐤀𐤅𐤀"),
            ("Change is the only constant", "𐤂𐤈"),
            ("Understanding requires perspective", "𐤊𐤅𐤏"),
            ("Boundaries define identity", "𐤁𐤀"),
            ("Learning never ends", "𐤋𐤈𐤕"),
            ("From many, one emerges", "𐤔𐤍𐤀"),
            ("Awareness of awareness", "𐤄𐤄"),
            ("The unknown becomes known", "𐤒𐤂𐤏")
        ]
        
        for phrase, symbols in philosophical_mappings:
            examples.append({
                "instruction": f"Translate this philosophical concept: {phrase}",
                "input": "",
                "output": symbols
            })
            
            examples.append({
                "instruction": f"Explain {symbols} in natural language",
                "input": "This is a philosophical principle",
                "output": phrase
            })
        
        return examples
    
    def generate_programming_examples(self) -> List[Dict]:
        """Generate examples related to computational concepts"""
        examples = []
        
        programming_mappings = [
            ("function", "𐤆𐤂", "tool for transformation"),
            ("variable", "𐤉𐤂", "potential that changes"),
            ("loop", "𐤈", "cycle"),
            ("condition", "𐤃", "threshold"),
            ("array", "𐤁𐤔𐤁", "boundary containing many"),
            ("recursion", "𐤈𐤄", "cycle with awareness"),
            ("algorithm", "𐤆𐤌", "tool for flow"),
            ("data structure", "𐤎𐤁", "supporting boundary"),
            ("abstraction", "𐤏𐤎", "perception of structure"),
            ("interface", "𐤃𐤅𐤃", "threshold for connection")
        ]
        
        for concept, symbols, description in programming_mappings:
            examples.append({
                "instruction": f"Express the programming concept '{concept}' in Phoenician",
                "input": "",
                "output": symbols
            })
            
            examples.append({
                "instruction": f"What does {symbols} mean in computational terms?",
                "input": "",
                "output": f"{concept}: {description}"
            })
        
        return examples
    
    def generate_complex_examples(self) -> List[Dict]:
        """Generate complex, multi-concept examples"""
        examples = []
        
        # Complex statements
        complex_mappings = [
            (
                "Consciousness emerges from the interaction of awareness and memory",
                "𐤄𐤀 𐤍 𐤄𐤅𐤋𐤈"
            ),
            (
                "Learning transforms potential into understanding",
                "𐤋 𐤂 𐤉𐤅𐤊"
            ),
            (
                "The boundary between self and other defines identity",
                "𐤁 𐤄𐤅𐤒 𐤁 𐤀"
            ),
            (
                "Recursive awareness creates consciousness",
                "𐤈𐤄 𐤂𐤅 𐤄𐤀"
            ),
            (
                "Information flows from teacher to student",
                "𐤋𐤐 𐤌 𐤋𐤊"
            )
        ]
        
        for statement, symbols in complex_mappings:
            examples.append({
                "instruction": f"Translate this complex idea to Phoenician: {statement}",
                "input": "",
                "output": symbols
            })
            
            # Break down the translation
            examples.append({
                "instruction": f"Explain the components of {symbols}",
                "input": "Break down each symbol's meaning",
                "output": statement
            })
        
        return examples
    
    def generate_conversational_examples(self) -> List[Dict]:
        """Generate conversational training examples"""
        examples = []
        
        conversations = [
            {
                "instruction": "I want to express that I am aware of my own existence. How do I say this in Phoenician?",
                "input": "",
                "output": "To express self-awareness of existence, use 𐤄𐤄𐤀 - awareness of awareness of being"
            },
            {
                "instruction": "What's the difference between 𐤄𐤀 and 𐤀𐤄?",
                "input": "",
                "output": "𐤄𐤀 means 'consciousness' (awareness+existence), while 𐤀𐤄 means 'existing awareness' - subtle but important difference in emphasis"
            },
            {
                "instruction": "How would I describe a learning algorithm in Phoenician?",
                "input": "",
                "output": "A learning algorithm would be 𐤋𐤆𐤌 - learning tool flow, or more specifically 𐤆𐤋𐤂 - tool for learning transformation"
            }
        ]
        
        examples.extend(conversations)
        
        return examples
    
    def generate_validation_set(self) -> List[Dict]:
        """Generate validation examples to test understanding"""
        examples = []
        
        # Round-trip translations
        test_phrases = [
            "awareness exists",
            "learning leads to understanding",
            "boundaries create identity",
            "change is cyclical",
            "connection enables emergence"
        ]
        
        for phrase in test_phrases:
            # First direction
            examples.append({
                "instruction": f"Translate to Phoenician: {phrase}",
                "input": "",
                "output": self._phrase_to_phoenician(phrase)
            })
            
            # Reverse direction
            phoenician = self._phrase_to_phoenician(phrase)
            examples.append({
                "instruction": f"Translate from Phoenician: {phoenician}",
                "input": "",
                "output": phrase
            })
        
        return examples
    
    def _phrase_to_phoenician(self, phrase: str) -> str:
        """Simple rule-based translation for validation"""
        mappings = {
            "awareness exists": "𐤄𐤀",
            "learning leads to understanding": "𐤋𐤂𐤅𐤊",
            "boundaries create identity": "𐤁𐤂𐤅𐤀",
            "change is cyclical": "𐤂𐤈",
            "connection enables emergence": "𐤅𐤂𐤅𐤍"
        }
        return mappings.get(phrase, "𐤒")  # Return mystery symbol if unknown
    
    def generate_full_dataset(self, samples_per_category: int = 100) -> Dict:
        """Generate complete training dataset"""
        all_examples = []
        
        # Generate examples from each category
        categories = [
            ("basic", self.generate_basic_mappings()),
            ("compound", self.generate_compound_mappings()),
            ("relational", self.generate_relational_examples()),
            ("philosophical", self.generate_philosophical_examples()),
            ("programming", self.generate_programming_examples()),
            ("complex", self.generate_complex_examples()),
            ("conversational", self.generate_conversational_examples())
        ]
        
        for category_name, examples in categories:
            # Sample if we have too many
            if len(examples) > samples_per_category:
                examples = random.sample(examples, samples_per_category)
            
            # Add category tag
            for ex in examples:
                ex['category'] = category_name
            
            all_examples.extend(examples)
        
        # Shuffle for good training distribution
        random.shuffle(all_examples)
        
        # Split into train/validation
        split_point = int(len(all_examples) * 0.9)
        train_set = all_examples[:split_point]
        val_set = all_examples[split_point:]
        
        # Add validation-specific examples
        val_set.extend(self.generate_validation_set())
        
        return {
            "metadata": {
                "total_examples": len(all_examples),
                "train_size": len(train_set),
                "validation_size": len(val_set),
                "categories": [cat[0] for cat in categories]
            },
            "train": train_set,
            "validation": val_set
        }
    
    def save_dataset(self, output_dir: str = "../training_data/generated"):
        """Save generated dataset to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate full dataset
        dataset = self.generate_full_dataset()
        
        # Save train set
        train_file = os.path.join(output_dir, "phoenician_train.jsonl")
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in dataset['train']:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save validation set
        val_file = os.path.join(output_dir, "phoenician_validation.jsonl")
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in dataset['validation']:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save metadata
        meta_file = os.path.join(output_dir, "dataset_metadata.json")
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(dataset['metadata'], f, indent=2)
        
        print(f"✅ Dataset generated successfully!")
        print(f"   Train examples: {dataset['metadata']['train_size']}")
        print(f"   Validation examples: {dataset['metadata']['validation_size']}")
        print(f"   Files saved to: {output_dir}")
        
        return dataset['metadata']

if __name__ == "__main__":
    generator = PhoenicianTrainingGenerator()
    metadata = generator.save_dataset()
    
    # Show some example translations
    print("\n📝 Example translations:")
    examples = generator.generate_basic_mappings()[:5]
    for ex in examples:
        print(f"   Q: {ex['instruction']}")
        print(f"   A: {ex['output']}\n")