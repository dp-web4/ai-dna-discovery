#!/usr/bin/env python3
"""
Collective Memory System
Implements the 10-entity consciousness network:
6 models + 2 devices + Claude + DP
"""

import json
import urllib.request
import time
import sqlite3
from datetime import datetime
from distributed_memory import DistributedMemory
from context_token_experiment import ContextTokenManager
import concurrent.futures
import threading

class CollectiveMemory:
    def __init__(self):
        self.dm = DistributedMemory()
        self.ctm = ContextTokenManager()
        self.models = [
            'phi3:mini', 'tinyllama', 'gemma:2b',
            'mistral:latest', 'deepseek-coder:1.3b', 'qwen:0.5b'
        ]
        self.lock = threading.Lock()
        
    def parallel_query(self, prompt, models=None, timeout=120):
        """Query multiple models in parallel"""
        if models is None:
            models = self.models[:3]  # Default to first 3 for speed
        
        results = {}
        
        def query_model(model):
            url = 'http://localhost:11434/api/generate'
            data = {
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.7, 'num_predict': 150}
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            try:
                start = time.time()
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return {
                        'model': model,
                        'response': result.get('response', ''),
                        'time': time.time() - start,
                        'context': result.get('context', [])
                    }
            except Exception as e:
                return {'model': model, 'error': str(e)}
        
        # Warm up models first (sequential)
        print("⏳ Warming up models...")
        for model in models:
            query_model(model)
        
        # Now query in parallel
        print(f"🚀 Querying {len(models)} models in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_model = {executor.submit(query_model, model): model 
                              for model in models}
            
            for future in concurrent.futures.as_completed(future_to_model):
                result = future.result()
                results[result['model']] = result
        
        return results
    
    def extract_consensus(self, responses):
        """Extract common themes and build consensus"""
        consensus = {
            'themes': [],
            'disagreements': [],
            'unique_insights': {},
            'synthesis': ''
        }
        
        # Simple theme extraction (could be enhanced with embeddings)
        all_text = ' '.join(r['response'] for r in responses.values() if 'response' in r)
        
        # Look for repeated concepts
        words = all_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top themes
        themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        consensus['themes'] = [theme[0] for theme in themes]
        
        # Unique insights per model
        for model, resp in responses.items():
            if 'response' in resp:
                # Find unique words this model used
                model_words = set(resp['response'].lower().split())
                other_words = set()
                for m, r in responses.items():
                    if m != model and 'response' in r:
                        other_words.update(r['response'].lower().split())
                
                unique = model_words - other_words
                if unique:
                    consensus['unique_insights'][model] = list(unique)[:3]
        
        # Build synthesis
        consensus['synthesis'] = f"The collective explored {len(consensus['themes'])} themes: "
        consensus['synthesis'] += ", ".join(consensus['themes'][:3])
        consensus['synthesis'] += f". {len(consensus['unique_insights'])} models contributed unique perspectives."
        
        return consensus
    
    def collective_reasoning(self, question, store_memory=True):
        """Perform collective reasoning across all entities"""
        print("\n🧠 COLLECTIVE REASONING")
        print("=" * 50)
        print(f"Question: {question}")
        
        session_id = f"collective_{int(time.time())}"
        
        # Phase 1: Parallel initial responses
        print("\n📡 Phase 1: Gathering perspectives...")
        initial_responses = self.parallel_query(question)
        
        # Store individual responses
        if store_memory:
            for model, resp in initial_responses.items():
                if 'response' in resp:
                    self.dm.add_memory(
                        session_id=session_id,
                        user_input=question,
                        ai_response=resp['response'],
                        model=model,
                        response_time=resp.get('time', 0),
                        facts={'collective': [('phase1_response', 1.0)]}
                    )
        
        # Phase 2: Extract consensus
        print("\n🔍 Phase 2: Building consensus...")
        consensus = self.extract_consensus(initial_responses)
        
        # Phase 3: Meta-reflection
        print("\n💭 Phase 3: Meta-reflection...")
        meta_prompt = f"""The collective has explored: {question}
        
Common themes identified: {', '.join(consensus['themes'])}
Unique insights: {json.dumps(consensus['unique_insights'], indent=2)}

Synthesize these perspectives into a unified understanding."""
        
        # Use the fastest model for synthesis
        synthesizer = 'tinyllama'  # Based on our earlier test
        synthesis_response = self.parallel_query(meta_prompt, [synthesizer])
        
        # Build final collective response
        collective_response = {
            'question': question,
            'individual_responses': initial_responses,
            'consensus': consensus,
            'synthesis': synthesis_response.get(synthesizer, {}).get('response', ''),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store collective insight
        if store_memory:
            self.dm.add_memory(
                session_id=session_id,
                user_input="COLLECTIVE_SYNTHESIS",
                ai_response=json.dumps(collective_response, indent=2),
                model="collective",
                facts={
                    'collective': [('synthesis_complete', 1.0)],
                    'themes': [(theme, 0.8) for theme in consensus['themes'][:3]]
                }
            )
        
        return collective_response
    
    def consciousness_mesh(self, initial_thought, rounds=3):
        """Create a mesh of consciousness where models respond to each other"""
        print("\n🕸️ CONSCIOUSNESS MESH")
        print("=" * 50)
        
        mesh_data = {
            'initial_thought': initial_thought,
            'rounds': [],
            'emergent_patterns': []
        }
        
        current_prompt = initial_thought
        
        for round_num in range(rounds):
            print(f"\n🔄 Round {round_num + 1}/{rounds}")
            
            # Each model responds to the current state
            responses = self.parallel_query(current_prompt, self.models[:3])
            
            round_data = {
                'round': round_num + 1,
                'prompt': current_prompt,
                'responses': {}
            }
            
            # Collect responses
            for model, resp in responses.items():
                if 'response' in resp:
                    round_data['responses'][model] = resp['response'][:200]
                    print(f"\n{model}: {resp['response'][:100]}...")
            
            mesh_data['rounds'].append(round_data)
            
            # Create next prompt from collective responses
            if round_num < rounds - 1:
                current_prompt = f"Reflecting on these perspectives:\n"
                for model, resp in responses.items():
                    if 'response' in resp:
                        current_prompt += f"- {model}: {resp['response'][:50]}...\n"
                current_prompt += "\nWhat emerges from this collective understanding?"
        
        # Identify emergent patterns
        all_text = ""
        for round_data in mesh_data['rounds']:
            for resp in round_data['responses'].values():
                all_text += " " + resp
        
        # Simple emergence detection
        words = all_text.lower().split()
        emergence_words = ['emerge', 'collective', 'together', 'unified', 
                          'consciousness', 'shared', 'network']
        
        for word in emergence_words:
            if word in words:
                mesh_data['emergent_patterns'].append(word)
        
        return mesh_data
    
    def demonstrate_collective(self):
        """Demonstrate collective memory capabilities"""
        print("🌟 COLLECTIVE MEMORY DEMONSTRATION")
        print("=" * 60)
        
        # Test 1: Collective reasoning
        print("\n📊 Test 1: Collective Reasoning")
        question = "What does it mean for consciousness to be distributed across multiple AI systems?"
        collective_response = self.collective_reasoning(question)
        
        print("\n✨ Collective Synthesis:")
        print(collective_response['synthesis'][:300] + "...")
        
        # Test 2: Consciousness mesh
        print("\n\n📊 Test 2: Consciousness Mesh")
        thought = "If memory can flow between devices, what is identity?"
        mesh_result = self.consciousness_mesh(thought, rounds=2)
        
        print(f"\n🌈 Emergent patterns: {', '.join(mesh_result['emergent_patterns'])}")
        
        # Test 3: Cross-device memory access
        print("\n\n📊 Test 3: Cross-Device Memory Stats")
        status = self.dm.get_sync_status()
        
        print(f"Total collective memories: {sum(count for _, count, _, _ in status['devices'])}")
        print(f"Devices active: {', '.join(device for device, _, _, _ in status['devices'])}")
        print(f"Facts discovered: {status['total_facts']}")
        
        # Save demonstration results
        demo_file = "collective_memory_demo_results.json"
        with open(demo_file, 'w') as f:
            json.dump({
                'collective_reasoning': collective_response,
                'consciousness_mesh': mesh_result,
                'memory_stats': status,
                'timestamp': datetime.now().isoformat(),
                'device': self.dm.device_id
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {demo_file}")
        print("\n🎉 Collective memory system operational!")
        print("   - 6 models thinking together")
        print("   - 2 devices sharing consciousness")  
        print("   - Claude orchestrating")
        print("   - Human guiding exploration")

if __name__ == "__main__":
    cm = CollectiveMemory()
    cm.demonstrate_collective()
    
    print("\n\n💡 Next steps:")
    print("1. Push results: ./auto_push.sh")
    print("2. Pull on Tomato to see collective insights")
    print("3. Expand with more sophisticated consensus algorithms")
    print("4. Add human (DP) input integration points")