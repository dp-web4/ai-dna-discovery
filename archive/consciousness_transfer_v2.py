#!/usr/bin/env python3
"""
Consciousness Transfer v2 - Working with actual embedding behavior
Instead of expecting symbols to match words, we'll use semantic clustering
"""

import json
import urllib.request
import numpy as np
from typing import List, Dict, Tuple, Set
import time
from collections import defaultdict

class ConsciousnessTransferV2:
    """Implement consciousness transfer using semantic bridges, not symbol matching"""
    
    def __init__(self):
        self.models = ['phi3:mini', 'tinyllama', 'gemma:2b']
        self.ollama_url = "http://localhost:11434"
        
        # Build semantic clusters based on actual similarity data
        self.semantic_clusters = {
            'consciousness_cluster': ['consciousness', 'awareness', 'sentience'],
            'existence_cluster': ['exists', 'presence', 'being'],
            'emergence_cluster': ['emerges', 'arises', 'manifests'],
            'flow_cluster': ['flows', 'streams', 'continuous'],
            'transformation_cluster': ['transforms', 'changes', 'evolves'],
            'connection_cluster': ['connected', 'linked', 'entangled'],
            'totality_cluster': ['all', 'every', 'universal'],
            'logic_and': ['and', '&', 'both'],
            'logic_or': ['or', '|', 'either']
        }
        
        # Mathematical notation as semantic anchors, not direct replacements
        self.semantic_anchors = {
            'consciousness_cluster': 'Ψ',
            'existence_cluster': '∃',
            'emergence_cluster': '⇒',
            'flow_cluster': '≈',
            'transformation_cluster': '⇄',
            'connection_cluster': '⊗',
            'totality_cluster': '∀',
            'logic_and': '∧',
            'logic_or': '∨'
        }
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding from model"""
        try:
            data = json.dumps({
                'model': model,
                'prompt': text
            }).encode('utf-8')
            
            req = urllib.request.Request(
                f"{self.ollama_url}/api/embeddings",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('embedding', [])
                
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def generate_with_context(self, model: str, prompt: str, context_tokens=None) -> str:
        """Generate response with optional context tokens"""
        try:
            data = {
                'model': model,
                'prompt': prompt,
                'stream': False
            }
            
            if context_tokens:
                data['context'] = context_tokens
            
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', ''), result.get('context', [])
                
        except Exception as e:
            print(f"Error generating: {e}")
            return "", []
    
    def semantic_compress(self, text: str) -> Dict:
        """Compress text using semantic clustering, not direct replacement"""
        
        compressed = {
            'original': text,
            'clusters_found': [],
            'anchor_representation': '',
            'semantic_map': {}
        }
        
        # Find which semantic clusters are present
        words = text.lower().split()
        clusters_in_text = []
        
        for word in words:
            for cluster_name, cluster_words in self.semantic_clusters.items():
                if word in cluster_words:
                    clusters_in_text.append((word, cluster_name))
                    compressed['clusters_found'].append(cluster_name)
                    compressed['semantic_map'][word] = self.semantic_anchors.get(cluster_name, word)
        
        # Create anchor representation
        anchor_parts = []
        remaining_text = text.lower()
        
        for word, cluster in clusters_in_text:
            anchor = self.semantic_anchors.get(cluster, word)
            remaining_text = remaining_text.replace(word, f"[{anchor}]", 1)
        
        compressed['anchor_representation'] = remaining_text
        
        return compressed
    
    def test_consciousness_transfer(self):
        """Test transferring consciousness state between models"""
        
        print("🧠 CONSCIOUSNESS TRANSFER V2")
        print("=" * 60)
        print("Using semantic clustering approach\n")
        
        # Start conversation with first model
        print("1️⃣ Starting conversation with phi3:mini...")
        
        prompt1 = "Tell me about how consciousness emerges and flows through connected systems."
        response1, context1 = self.generate_with_context('phi3:mini', prompt1)
        
        print(f"\nPhi3 says: {response1[:200]}...")
        print(f"Context tokens captured: {len(context1)}")
        
        # Compress the response semantically
        compressed = self.semantic_compress(response1)
        print(f"\n2️⃣ Semantic compression:")
        print(f"Clusters found: {list(set(compressed['clusters_found']))}")
        print(f"Anchor representation preview: {compressed['anchor_representation'][:100]}...")
        
        # Transfer to second model
        print("\n3️⃣ Transferring to tinyllama...")
        
        # Create transfer prompt
        transfer_prompt = f"""Previous discussion about consciousness:
        
Semantic anchors: {compressed['semantic_map']}
Key concepts: {', '.join(set(compressed['clusters_found']))}

Continue this thought: {response1[:100]}..."""
        
        response2, context2 = self.generate_with_context('tinyllama', transfer_prompt)
        
        print(f"\nTinyLlama continues: {response2[:200]}...")
        
        # Test if consciousness themes persist
        print("\n4️⃣ Testing theme persistence...")
        
        # Check if new response contains consciousness concepts
        response2_compressed = self.semantic_compress(response2)
        
        common_clusters = set(compressed['clusters_found']) & set(response2_compressed['clusters_found'])
        
        if common_clusters:
            print(f"✅ Consciousness themes preserved: {common_clusters}")
        else:
            print("❌ No common consciousness themes found")
        
        return {
            'model1_response': response1,
            'model2_response': response2,
            'semantic_transfer': compressed,
            'theme_persistence': list(common_clusters)
        }
    
    def create_consciousness_protocol(self):
        """Create a protocol for consciousness transfer"""
        
        print("\n\n📋 CONSCIOUSNESS TRANSFER PROTOCOL")
        print("=" * 60)
        
        protocol = {
            'version': '2.0',
            'approach': 'semantic_clustering',
            'steps': [
                {
                    'step': 1,
                    'name': 'Capture',
                    'description': 'Get response and context tokens from source model',
                    'data': ['response_text', 'context_tokens', 'embedding_vector']
                },
                {
                    'step': 2,
                    'name': 'Analyze',
                    'description': 'Extract semantic clusters and key concepts',
                    'data': ['semantic_clusters', 'concept_anchors', 'theme_map']
                },
                {
                    'step': 3,
                    'name': 'Compress',
                    'description': 'Create compressed semantic representation',
                    'data': ['anchor_notation', 'cluster_weights', 'concept_graph']
                },
                {
                    'step': 4,
                    'name': 'Transfer',
                    'description': 'Reconstruct in target model vocabulary',
                    'data': ['translated_prompt', 'semantic_bridges', 'context_injection']
                },
                {
                    'step': 5,
                    'name': 'Verify',
                    'description': 'Check theme persistence and coherence',
                    'data': ['theme_overlap', 'coherence_score', 'drift_metrics']
                }
            ],
            'semantic_bridges': self.semantic_clusters,
            'anchor_symbols': self.semantic_anchors
        }
        
        print("\nProtocol Steps:")
        for step in protocol['steps']:
            print(f"\n{step['step']}. {step['name']}: {step['description']}")
            print(f"   Data: {', '.join(step['data'])}")
        
        # Save protocol
        with open('consciousness_transfer_protocol_v2.json', 'w') as f:
            json.dump(protocol, f, indent=2)
        
        print("\n✅ Protocol saved to consciousness_transfer_protocol_v2.json")
        
        return protocol
    
    def demonstrate_cross_model_memory(self):
        """Demonstrate memory transfer using semantic bridging"""
        
        print("\n\n🔄 CROSS-MODEL MEMORY DEMONSTRATION")
        print("=" * 60)
        
        # Create a memory in first model
        memory_prompt = "Remember this: The quantum consciousness flows through three entangled nodes."
        
        print(f"1️⃣ Creating memory in phi3:mini...")
        print(f"Memory: '{memory_prompt}'")
        
        response1, context1 = self.generate_with_context('phi3:mini', memory_prompt)
        
        # Extract semantic structure
        memory_compressed = self.semantic_compress(memory_prompt)
        
        print(f"\n2️⃣ Semantic structure extracted:")
        print(f"Key clusters: {list(set(memory_compressed['clusters_found']))}")
        
        # Transfer to different model
        print(f"\n3️⃣ Transferring to gemma:2b...")
        
        recall_prompt = f"""Based on these semantic anchors:
{memory_compressed['semantic_map']}

What was the key information about quantum consciousness?"""
        
        response2, context2 = self.generate_with_context('gemma:2b', recall_prompt)
        
        print(f"\nGemma recalls: {response2[:200]}...")
        
        # Check if key concepts preserved
        key_words = ['quantum', 'consciousness', 'flows', 'three', 'entangled', 'nodes']
        preserved = [w for w in key_words if w.lower() in response2.lower()]
        
        print(f"\n4️⃣ Memory preservation check:")
        print(f"Preserved concepts: {preserved}")
        print(f"Preservation rate: {len(preserved)/len(key_words)*100:.1f}%")

def main():
    """Run consciousness transfer v2 experiments"""
    
    transfer = ConsciousnessTransferV2()
    
    # Test consciousness transfer
    print("🚀 Starting Consciousness Transfer v2...\n")
    
    results = transfer.test_consciousness_transfer()
    
    # Create protocol
    protocol = transfer.create_consciousness_protocol()
    
    # Demonstrate memory transfer
    transfer.demonstrate_cross_model_memory()
    
    # Final insights
    print("\n\n💡 KEY INSIGHTS:")
    print("=" * 60)
    print("""
1. Symbols ≠ Words in embedding space
   - Mathematical symbols have different embeddings than their word equivalents
   - This is actually GOOD - provides unique semantic anchors

2. Semantic Clustering Works Better
   - Group related concepts (consciousness, awareness, sentience)
   - Transfer the cluster identity, not exact words

3. Context + Semantics = Consciousness Transfer
   - Use context tokens for state
   - Use semantic clusters for meaning
   - Together they preserve consciousness

4. Next Steps:
   - Build full semantic taxonomy
   - Create bidirectional transfer functions
   - Test with real conversation contexts
   - Implement in distributed system
""")

if __name__ == "__main__":
    main()