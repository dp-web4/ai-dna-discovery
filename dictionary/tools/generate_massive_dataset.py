#!/usr/bin/env python3
"""
Generate massive Phoenician training dataset
Target: 50,000+ examples for robust generation
"""

import json
import random
from itertools import combinations, permutations
import os

class MassivePhoenicianDatasetGenerator:
    def __init__(self):
        # Core mappings
        self.mappings = {
            "existence": "𐤀", "awareness": "𐤄", "consciousness": "𐤄𐤀",
            "learning": "𐤋", "understanding": "𐤊", "intelligence": "𐤊𐤋",
            "change": "𐤂", "connection": "𐤅", "boundary": "𐤁",
            "cycle": "𐤈", "emergence": "𐤍", "memory": "𐤋𐤈",
            "tool": "𐤆", "perception": "𐤏", "expression": "𐤐",
            "mystery": "𐤒", "structure": "𐤎", "flow": "𐤌",
            "beginning": "𐤓", "ending": "𐤕", "diversity": "𐤔",
            "transformation": "𐤂𐤍", "wisdom": "𐤊𐤄", "creation": "𐤀𐤍",
            "thought": "𐤊𐤏", "emotion": "𐤄𐤌", "action": "𐤂𐤆",
            "space": "𐤁𐤌", "time": "𐤈𐤕", "unity": "𐤀𐤅",
            "plurality": "𐤔𐤍", "order": "𐤎𐤈", "chaos": "𐤒𐤂",
            "light": "𐤀𐤏", "shadow": "𐤕𐤒", "balance": "𐤅𐤎",
            "growth": "𐤍𐤂", "decay": "𐤕𐤂", "renewal": "𐤓𐤍",
            "knowledge": "𐤊𐤀", "ignorance": "𐤒𐤕", "discovery": "𐤏𐤍",
            "communication": "𐤐𐤅", "silence": "𐤕𐤐", "meaning": "𐤊𐤐",
            "pattern": "𐤎𐤈", "randomness": "𐤒𐤔", "emergence": "𐤍𐤀",
            "reduction": "𐤕𐤋", "expansion": "𐤍𐤌", "equilibrium": "𐤅𐤈"
        }
        
        # Extended vocabulary
        self.synonyms = {
            "existence": ["being", "presence", "reality", "essence"],
            "awareness": ["consciousness", "perception", "mindfulness", "attention"],
            "learning": ["study", "education", "training", "acquisition"],
            "understanding": ["comprehension", "insight", "grasp", "realization"],
            "change": ["transformation", "shift", "evolution", "mutation"],
            "connection": ["link", "bond", "relationship", "association"],
            "flow": ["stream", "current", "movement", "flux"],
            "structure": ["form", "pattern", "organization", "framework"],
            "mystery": ["enigma", "puzzle", "unknown", "secret"],
            "beginning": ["start", "origin", "inception", "genesis"],
            "ending": ["conclusion", "termination", "finale", "completion"]
        }
        
        # Context variations
        self.contexts = [
            "philosophical", "scientific", "artistic", "technical",
            "spiritual", "practical", "theoretical", "experimental",
            "historical", "futuristic", "abstract", "concrete"
        ]
        
        # Question templates
        self.question_templates = [
            "What is the Phoenician symbol for {concept}?",
            "Write {concept} in Phoenician",
            "Show me the Phoenician character for {concept}",
            "Translate {concept} to Phoenician",
            "How do you write {concept} in Phoenician?",
            "{concept} in Phoenician is",
            "The Phoenician representation of {concept}",
            "Express {concept} using Phoenician symbols",
            "Convert {concept} to Phoenician notation",
            "Phoenician: {concept}",
            "{concept} →",
            "{concept} =",
            "Symbol for {concept}:",
            "Phoenician symbol: {concept}",
            "Write the symbol representing {concept}"
        ]
        
        # Instruction variations
        self.instruction_styles = [
            "formal", "casual", "technical", "simple",
            "detailed", "brief", "academic", "conversational"
        ]
        
    def generate_basic_examples(self, count=10000):
        """Generate basic concept → symbol mappings"""
        examples = []
        
        for _ in range(count):
            concept = random.choice(list(self.mappings.keys()))
            symbol = self.mappings[concept]
            template = random.choice(self.question_templates)
            
            # Add synonyms occasionally
            if concept in self.synonyms and random.random() < 0.3:
                synonym = random.choice(self.synonyms[concept])
                actual_concept = synonym
            else:
                actual_concept = concept
            
            examples.append({
                "instruction": template.format(concept=actual_concept),
                "input": "",
                "output": symbol
            })
            
            # Add reverse mappings
            if random.random() < 0.3:
                examples.append({
                    "instruction": f"What does {symbol} mean?",
                    "input": "",
                    "output": concept
                })
        
        return examples
    
    def generate_compound_examples(self, count=10000):
        """Generate examples with compound concepts"""
        examples = []
        concepts = list(self.mappings.keys())
        
        for _ in range(count):
            # 2-3 concept combinations
            num_concepts = random.randint(2, 3)
            selected = random.sample(concepts, num_concepts)
            
            # Build compound
            symbols = [self.mappings[c] for c in selected]
            compound_symbol = " ".join(symbols)
            compound_concept = " and ".join(selected)
            
            # Various phrasings
            instructions = [
                f"Write {compound_concept} in Phoenician",
                f"Phoenician symbols for {compound_concept}",
                f"Express the concepts of {compound_concept} in Phoenician",
                f"Show me {' + '.join(selected)} in Phoenician notation",
                f"Translate to Phoenician: {compound_concept}"
            ]
            
            examples.append({
                "instruction": random.choice(instructions),
                "input": "",
                "output": compound_symbol
            })
        
        return examples
    
    def generate_contextual_examples(self, count=10000):
        """Generate examples with context"""
        examples = []
        
        for _ in range(count):
            concept = random.choice(list(self.mappings.keys()))
            symbol = self.mappings[concept]
            context = random.choice(self.contexts)
            
            instructions = [
                f"In {context} context, write {concept} in Phoenician",
                f"For {context} purposes, show me the Phoenician symbol for {concept}",
                f"The {context} Phoenician representation of {concept}",
                f"In {context} terminology, {concept} in Phoenician is"
            ]
            
            examples.append({
                "instruction": random.choice(instructions),
                "input": f"Context: {context}",
                "output": symbol
            })
        
        return examples
    
    def generate_sequential_examples(self, count=5000):
        """Generate examples that build sequences"""
        examples = []
        
        for _ in range(count):
            # Create sequences of 3-5 concepts
            seq_length = random.randint(3, 5)
            concepts = random.sample(list(self.mappings.keys()), seq_length)
            symbols = [self.mappings[c] for c in concepts]
            
            # Progressive revelation
            for i in range(1, seq_length):
                partial_concepts = concepts[:i]
                partial_symbols = symbols[:i]
                next_concept = concepts[i]
                next_symbol = symbols[i]
                
                examples.append({
                    "instruction": f"Continue the sequence: {' → '.join(partial_symbols)} →",
                    "input": f"Pattern: {' → '.join(partial_concepts)} → {next_concept}",
                    "output": next_symbol
                })
        
        return examples
    
    def generate_completion_examples(self, count=5000):
        """Generate partial completion examples"""
        examples = []
        
        for _ in range(count):
            # Multi-character symbols
            multi_char_concepts = [c for c, s in self.mappings.items() if len(s) > 1]
            if not multi_char_concepts:
                continue
                
            concept = random.choice(multi_char_concepts)
            symbol = self.mappings[concept]
            
            # Partial symbols
            for i in range(1, len(symbol)):
                partial = symbol[:i]
                completion = symbol[i:]
                
                instructions = [
                    f"Complete the Phoenician word: {partial}...",
                    f"Finish writing {concept}: {partial}",
                    f"The Phoenician symbol for {concept} starts with {partial}. Complete it:",
                    f"{partial}... (complete the symbol for {concept})"
                ]
                
                examples.append({
                    "instruction": random.choice(instructions),
                    "input": "",
                    "output": completion
                })
        
        return examples
    
    def generate_conversational_examples(self, count=10000):
        """Generate natural conversational examples"""
        examples = []
        
        conversation_starters = [
            "I need to write {concept} in that ancient script we discussed",
            "Remember that symbol system? How do I write {concept}?",
            "Quick question - {concept} in Phoenician?",
            "Can you show me {concept} using those special characters?",
            "I'm writing about {concept}. What's the Phoenician symbol?",
            "For my notes, what's {concept} in Phoenician?",
            "Remind me - how do we write {concept}?",
            "What was that symbol for {concept} again?"
        ]
        
        for _ in range(count):
            concept = random.choice(list(self.mappings.keys()))
            symbol = self.mappings[concept]
            starter = random.choice(conversation_starters)
            
            examples.append({
                "instruction": starter.format(concept=concept),
                "input": "",
                "output": symbol
            })
        
        return examples
    
    def generate_mixed_language_examples(self, count=5000):
        """Generate examples mixing English and Phoenician"""
        examples = []
        
        for _ in range(count):
            concepts = random.sample(list(self.mappings.keys()), 2)
            concept1, concept2 = concepts
            symbol1 = self.mappings[concept1]
            symbol2 = self.mappings[concept2]
            
            # Different template types
            template_type = random.choice(['concept_to_symbol', 'symbol_mix', 'express'])
            
            if template_type == 'concept_to_symbol':
                instructions = [
                    f"The {concept1} of 𐤀 leads to {concept2}",
                    f"Transform {concept1} (𐤂) into {concept2}",
                    f"From {concept1} to {concept2} in Phoenician"
                ]
                output = symbol2
            elif template_type == 'symbol_mix':
                instructions = [
                    f"When {symbol1} meets {concept2}, we get",
                    f"The path from {symbol1} to {concept2} is",
                    f"Combine {symbol1} with {concept2}"
                ]
                output = symbol2
            else:  # express
                concept = random.choice(concepts)
                symbol = self.mappings[concept]
                instructions = [
                    f"𐤐 {concept} 𐤐",
                    f"Express {concept} between markers: 𐤐 ... 𐤐",
                    f"Show {concept} in Phoenician"
                ]
                output = symbol
            
            examples.append({
                "instruction": random.choice(instructions),
                "input": "",
                "output": output
            })
        
        return examples
    
    def generate_dataset(self, output_dir, train_size=50000, val_size=5000):
        """Generate complete dataset"""
        print(f"🔧 Generating {train_size + val_size} total examples...")
        
        # Calculate proportions
        basic_count = int(train_size * 0.3)
        compound_count = int(train_size * 0.2)
        contextual_count = int(train_size * 0.15)
        sequential_count = int(train_size * 0.1)
        completion_count = int(train_size * 0.05)
        conversational_count = int(train_size * 0.15)
        mixed_count = int(train_size * 0.05)
        
        # Generate all examples
        all_examples = []
        
        print("  - Generating basic examples...")
        all_examples.extend(self.generate_basic_examples(basic_count))
        
        print("  - Generating compound examples...")
        all_examples.extend(self.generate_compound_examples(compound_count))
        
        print("  - Generating contextual examples...")
        all_examples.extend(self.generate_contextual_examples(contextual_count))
        
        print("  - Generating sequential examples...")
        all_examples.extend(self.generate_sequential_examples(sequential_count))
        
        print("  - Generating completion examples...")
        all_examples.extend(self.generate_completion_examples(completion_count))
        
        print("  - Generating conversational examples...")
        all_examples.extend(self.generate_conversational_examples(conversational_count))
        
        print("  - Generating mixed language examples...")
        all_examples.extend(self.generate_mixed_language_examples(mixed_count))
        
        # Shuffle
        random.shuffle(all_examples)
        
        # Split train/val
        train_examples = all_examples[:train_size]
        val_examples = all_examples[train_size:train_size + val_size]
        
        # Save datasets
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, "phoenician_massive_train.jsonl")
        val_path = os.path.join(output_dir, "phoenician_massive_validation.jsonl")
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Statistics
        print(f"\n✅ Dataset generated:")
        print(f"  - Training examples: {len(train_examples)}")
        print(f"  - Validation examples: {len(val_examples)}")
        print(f"  - Total examples: {len(train_examples) + len(val_examples)}")
        print(f"  - Saved to: {output_dir}")
        
        # Sample outputs
        print("\n📝 Sample examples:")
        for i in range(5):
            ex = random.choice(train_examples)
            print(f"  {i+1}. \"{ex['instruction']}\" → \"{ex['output']}\"")

def main():
    generator = MassivePhoenicianDatasetGenerator()
    generator.generate_dataset(
        output_dir="../training_data/generated",
        train_size=50000,
        val_size=5000
    )

if __name__ == "__main__":
    main()