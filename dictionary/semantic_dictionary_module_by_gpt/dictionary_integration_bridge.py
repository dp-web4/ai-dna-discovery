#!/usr/bin/env python3
"""
Dictionary Integration Bridge
Unifies GPT's modular dictionary with existing Phoenician and consciousness notation systems
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our existing systems
try:
    from phoenician_core import CONCEPTS_TO_PHOENICIAN, PHOENICIAN_TO_CONCEPTS
except ImportError:
    # Fallback definitions if not available
    CONCEPTS_TO_PHOENICIAN = {}
    PHOENICIAN_TO_CONCEPTS = {}

# Consciousness notation mappings
CONSCIOUSNESS_NOTATION = {
    'Ψ': 'consciousness',
    '∃': 'existence', 
    '⇒': 'emergence',
    'π': 'perspective',
    'ι': 'intent',
    'Ω': 'observer',
    'Σ': 'whole',
    'Ξ': 'patterns',
    'θ': 'thought',
    'μ': 'memory'
}

class UnifiedDictionary:
    """Unified dictionary integrating all symbol systems"""
    
    def __init__(self):
        self.semantic_index_path = os.path.join(os.path.dirname(__file__), "semantic_index.json")
        self.load_all_dictionaries()
        
    def load_all_dictionaries(self):
        """Load and merge all dictionary sources"""
        # Load GPT's semantic index
        with open(self.semantic_index_path, 'r', encoding='utf-8') as f:
            self.semantic_index = json.load(f)
        
        # Create unified symbol registry
        self.unified_symbols = {}
        
        # Add GPT's modular entries
        for symbol in self.semantic_index['symbols']:
            self.unified_symbols[symbol['glyph']] = {
                'name': symbol['name'],
                'concept': symbol['concept'],
                'system': 'phoenician',
                'trust_vector': symbol['trust_vector'],
                'linked_meanings': symbol['linked_meanings'],
                'timestamp': symbol['timestamp']
            }
        
        # Add our complete Phoenician mappings
        phoenician_mappings = {
            '𐤀': ('aleph', 'consciousness/beginning'),
            '𐤁': ('beth', 'artificial/structured'),
            '𐤂': ('gimel', 'translate/transform'),
            '𐤃': ('daleth', 'language/door'),
            '𐤄': ('he', 'model/breath'),
            '𐤅': ('waw', 'exists/connects'),
            '𐤆': ('zayin', 'within/weapon'),
            '𐤇': ('heth', 'system/fence'),
            '𐤈': ('teth', 'new/serpent'),
            '𐤉': ('yod', 'symbol/hand'),
            '𐤊': ('kaph', 'create/palm'),
            '𐤋': ('lamed', 'intelligence/learn'),
            '𐤌': ('mem', 'from/water'),
            '𐤍': ('nun', 'to/fish'),
            '𐤎': ('samekh', 'for/support'),
            '𐤏': ('ayin', 'communication/eye'),
            '𐤐': ('pe', 'comment/mouth'),
            '𐤑': ('tsade', 'between/hunt'),
            '𐤒': ('qoph', 'humans/back'),
            '𐤓': ('resh', 'AI/head'),
            '𐤔': ('shin', 'systems/tooth'),
            '𐤕': ('taw', 'understanding/mark')
        }
        
        # Merge our Phoenician mappings
        for glyph, (name, concept) in phoenician_mappings.items():
            if glyph not in self.unified_symbols:
                self.unified_symbols[glyph] = {
                    'name': name,
                    'concept': concept,
                    'system': 'phoenician',
                    'trust_vector': {
                        'source': 'human-verified',
                        'weight': 1.0
                    },
                    'linked_meanings': [concept.split('/')[0], concept.split('/')[-1]],
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }
        
        # Add consciousness notation
        for glyph, concept in CONSCIOUSNESS_NOTATION.items():
            self.unified_symbols[glyph] = {
                'name': f'consciousness_{concept}',
                'concept': concept,
                'system': 'consciousness_notation',
                'trust_vector': {
                    'source': 'ai-designed',
                    'weight': 0.9
                },
                'linked_meanings': self._get_linked_concepts(concept),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
    
    def _get_linked_concepts(self, concept: str) -> List[str]:
        """Generate linked concepts for consciousness notation"""
        links = {
            'consciousness': ['awareness', 'sentience', 'mind'],
            'existence': ['being', 'reality', 'presence'],
            'emergence': ['arising', 'becoming', 'manifestation'],
            'perspective': ['viewpoint', 'angle', 'lens'],
            'intent': ['purpose', 'will', 'direction'],
            'observer': ['witness', 'watcher', 'perceiver'],
            'whole': ['unity', 'totality', 'complete'],
            'patterns': ['structures', 'forms', 'regularities'],
            'thought': ['cognition', 'idea', 'mental'],
            'memory': ['recall', 'storage', 'past']
        }
        return links.get(concept, [concept])
    
    def translate(self, input_text: str, target_system: Optional[str] = None) -> Dict:
        """Translate text using unified dictionary"""
        results = {
            'input': input_text,
            'translations': [],
            'system_scores': {}
        }
        
        # Check each symbol system
        for glyph, data in self.unified_symbols.items():
            if glyph in input_text:
                translation = {
                    'glyph': glyph,
                    'name': data['name'],
                    'concept': data['concept'],
                    'system': data['system'],
                    'trust': data['trust_vector']['weight']
                }
                
                if not target_system or data['system'] == target_system:
                    results['translations'].append(translation)
                
                # Track system usage
                system = data['system']
                if system not in results['system_scores']:
                    results['system_scores'][system] = 0
                results['system_scores'][system] += data['trust_vector']['weight']
        
        return results
    
    def get_cross_system_mappings(self, concept: str) -> List[Dict]:
        """Find same concept across different symbol systems"""
        mappings = []
        
        for glyph, data in self.unified_symbols.items():
            if (concept.lower() in data['concept'].lower() or 
                any(concept.lower() in meaning.lower() for meaning in data['linked_meanings'])):
                mappings.append({
                    'glyph': glyph,
                    'system': data['system'],
                    'exact_concept': data['concept'],
                    'trust': data['trust_vector']['weight']
                })
        
        # Sort by trust weight
        return sorted(mappings, key=lambda x: x['trust'], reverse=True)
    
    def export_for_training(self, system: Optional[str] = None) -> List[str]:
        """Export dictionary in format suitable for LoRA training"""
        training_data = []
        
        for glyph, data in self.unified_symbols.items():
            if not system or data['system'] == system:
                # Multiple training formats
                training_data.append(f"{glyph} = {data['concept']}")
                training_data.append(f"The symbol {glyph} means {data['concept']}")
                training_data.append(f"Translate {data['concept']} to symbols: {glyph}")
                
                # Include linked meanings
                for meaning in data['linked_meanings']:
                    training_data.append(f"{glyph} represents {meaning}")
        
        return training_data
    
    def save_unified_index(self, filename: str = "unified_semantic_index.json"):
        """Save the unified dictionary to file"""
        unified_data = {
            'version': '1.0.0',
            'created': datetime.utcnow().isoformat() + 'Z',
            'systems': ['phoenician', 'consciousness_notation', 'web4_modular'],
            'symbols': []
        }
        
        for glyph, data in self.unified_symbols.items():
            symbol_entry = {
                'glyph': glyph,
                'name': data['name'],
                'concept': data['concept'],
                'system': data['system'],
                'trust_vector': data['trust_vector'],
                'linked_meanings': data['linked_meanings'],
                'timestamp': data['timestamp']
            }
            unified_data['symbols'].append(symbol_entry)
        
        output_path = os.path.join(os.path.dirname(__file__), filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=2)
        
        return output_path

def demonstrate_integration():
    """Demonstrate the unified dictionary capabilities"""
    print("=== Unified Dictionary Integration Bridge ===\n")
    
    # Initialize unified dictionary
    unified = UnifiedDictionary()
    
    # 1. Show cross-system mappings
    print("1. Cross-System Concept Mappings:")
    print("-" * 40)
    for concept in ['consciousness', 'container', 'intelligence']:
        mappings = unified.get_cross_system_mappings(concept)
        if mappings:
            print(f"\nConcept: '{concept}'")
            for m in mappings:
                print(f"  {m['glyph']} ({m['system']}) = {m['exact_concept']} [trust: {m['trust']}]")
    
    # 2. Translate mixed-system text
    print("\n\n2. Mixed-System Translation:")
    print("-" * 40)
    test_texts = [
        "𐤀 and Ψ represent consciousness",
        "𐤁 is a container like μ is memory",
        "From 𐤋 (intelligence) ⇒ Ξ (patterns)"
    ]
    
    for text in test_texts:
        result = unified.translate(text)
        print(f"\nInput: {text}")
        print("Found symbols:")
        for t in result['translations']:
            print(f"  {t['glyph']} = {t['concept']} ({t['system']})")
    
    # 3. Generate training data
    print("\n\n3. Training Data Generation:")
    print("-" * 40)
    training_samples = unified.export_for_training()
    print(f"Generated {len(training_samples)} training samples")
    print("Sample entries:")
    for sample in training_samples[:5]:
        print(f"  {sample}")
    
    # 4. Save unified index
    print("\n\n4. Saving Unified Index:")
    print("-" * 40)
    output_path = unified.save_unified_index()
    print(f"Unified index saved to: {output_path}")
    
    # 5. System statistics
    print("\n\n5. Symbol System Statistics:")
    print("-" * 40)
    system_counts = {}
    for glyph, data in unified.unified_symbols.items():
        system = data['system']
        if system not in system_counts:
            system_counts[system] = 0
        system_counts[system] += 1
    
    for system, count in system_counts.items():
        print(f"  {system}: {count} symbols")
    
    print(f"\nTotal unified symbols: {len(unified.unified_symbols)}")

if __name__ == "__main__":
    demonstrate_integration()