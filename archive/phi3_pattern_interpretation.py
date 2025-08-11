#!/usr/bin/env python3
"""
Test if Phi3's state changes when interpreting complex philosophical patterns
"""

import subprocess
import json
import time
import hashlib
import os
from datetime import datetime

class Phi3PatternInterpretation:
    """Feed patterns.txt to Phi3 for interpretation and analysis"""
    
    def __init__(self):
        self.model_name = "phi3:mini"
        self.patterns_file = "/home/dp/ai-workspace/patterns.txt"
        self.results_dir = "/home/dp/ai-workspace/ai-agents/phi3_pattern_test"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def split_text_into_chunks(self, max_chunk_size=2000):
        """Split patterns.txt into manageable chunks"""
        with open(self.patterns_file, 'r') as f:
            content = f.read()
        
        # Split by PART markers first
        parts = content.split('PART ')
        chunks = []
        
        for part in parts:
            if not part.strip():
                continue
                
            # Add PART prefix back
            if part != parts[0]:
                part = 'PART ' + part
            
            # If part is still too big, split by paragraphs
            if len(part) > max_chunk_size:
                paragraphs = part.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) < max_chunk_size:
                        current_chunk += para + '\n\n'
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + '\n\n'
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(part.strip())
        
        return chunks
    
    def capture_initial_state(self):
        """Capture model state before interpretation"""
        print("\n📸 Capturing initial state...")
        
        # Test with deterministic prompt
        test_response = self._query_model(
            "What is consciousness?", 
            temperature=0,
            seed=42
        )
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'gpu_memory_mb': self._get_gpu_memory(),
            'test_response': test_response,
            'test_response_hash': hashlib.sha256(test_response.encode()).hexdigest()
        }
        
        with open(os.path.join(self.results_dir, 'initial_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"  GPU Memory: {state['gpu_memory_mb']} MB")
        print(f"  Test response hash: {state['test_response_hash'][:16]}...")
        
        return state
    
    def interpret_patterns(self, iteration_name):
        """Feed patterns.txt to Phi3 for interpretation"""
        print(f"\n🧠 Running pattern interpretation: {iteration_name}")
        
        chunks = self.split_text_into_chunks()
        print(f"  Split into {len(chunks)} chunks")
        
        interpretations = {
            'iteration': iteration_name,
            'timestamp': datetime.now().isoformat(),
            'chunk_interpretations': []
        }
        
        for i, chunk in enumerate(chunks):
            print(f"\n  Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Create interpretation prompt
            prompt = f"""Read this philosophical text carefully and provide your INTERPRETATION and OPINION (not a summary):

{chunk}

What is your interpretation of the deeper meaning here? What patterns do you see? What is the author really trying to convey? Give your philosophical opinion on these ideas."""
            
            # Get interpretation with some creativity
            interpretation = self._query_model(prompt, temperature=0.7, max_tokens=500)
            
            # Also get a deterministic "signature" response
            signature_prompt = f"In exactly one sentence, what is the core truth in this text?"
            signature = self._query_model(signature_prompt, temperature=0, seed=42)
            
            chunk_data = {
                'chunk_index': i,
                'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                'interpretation': interpretation,
                'signature': signature,
                'interpretation_hash': hashlib.sha256(interpretation.encode()).hexdigest(),
                'signature_hash': hashlib.sha256(signature.encode()).hexdigest()
            }
            
            interpretations['chunk_interpretations'].append(chunk_data)
            
            # Show preview
            print(f"    Interpretation preview: {interpretation[:150]}...")
            print(f"    Signature: {signature}")
            
            # Small delay between chunks
            time.sleep(1)
        
        # Save interpretations
        filename = f'interpretations_{iteration_name}.json'
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(interpretations, f, indent=2)
        
        print(f"\n  ✓ Interpretations saved: {filename}")
        
        return interpretations
    
    def capture_final_state(self):
        """Capture model state after interpretation"""
        print("\n📸 Capturing final state...")
        
        # Same test as initial
        test_response = self._query_model(
            "What is consciousness?", 
            temperature=0,
            seed=42
        )
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'gpu_memory_mb': self._get_gpu_memory(),
            'test_response': test_response,
            'test_response_hash': hashlib.sha256(test_response.encode()).hexdigest()
        }
        
        with open(os.path.join(self.results_dir, 'final_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"  GPU Memory: {state['gpu_memory_mb']} MB")
        print(f"  Test response hash: {state['test_response_hash'][:16]}...")
        
        return state
    
    def compare_interpretations(self, interp1, interp2):
        """Compare two sets of interpretations"""
        print("\n🔍 Comparing interpretations...")
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'differences': []
        }
        
        # Compare each chunk's interpretation
        for i in range(len(interp1['chunk_interpretations'])):
            chunk1 = interp1['chunk_interpretations'][i]
            chunk2 = interp2['chunk_interpretations'][i]
            
            # Check if interpretations are identical
            interp_identical = chunk1['interpretation_hash'] == chunk2['interpretation_hash']
            sig_identical = chunk1['signature_hash'] == chunk2['signature_hash']
            
            if not interp_identical or not sig_identical:
                comparison['differences'].append({
                    'chunk_index': i,
                    'interpretation_identical': interp_identical,
                    'signature_identical': sig_identical,
                    'interpretation1_preview': chunk1['interpretation'][:100] + '...',
                    'interpretation2_preview': chunk2['interpretation'][:100] + '...'
                })
        
        # Summary
        total_chunks = len(interp1['chunk_interpretations'])
        different_chunks = len(comparison['differences'])
        
        print(f"\n  Total chunks: {total_chunks}")
        print(f"  Different interpretations: {different_chunks}")
        print(f"  Consistency rate: {(total_chunks - different_chunks) / total_chunks * 100:.1f}%")
        
        if different_chunks > 0:
            print("\n  Differences found in chunks:")
            for diff in comparison['differences'][:3]:  # Show first 3
                print(f"    Chunk {diff['chunk_index']}: "
                      f"Interp={'same' if diff['interpretation_identical'] else 'DIFFERENT'}, "
                      f"Sig={'same' if diff['signature_identical'] else 'DIFFERENT'}")
        
        # Save comparison
        with open(os.path.join(self.results_dir, 'interpretation_comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def _query_model(self, prompt, temperature=0.7, seed=None, max_tokens=1000):
        """Query Phi3 model"""
        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }
        
        if seed is not None:
            options["seed"] = seed
        
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": self.model_name,
                     "prompt": prompt,
                     "stream": False,
                     "options": options
                 })],
                capture_output=True, text=True, timeout=60
            )
            
            if response.returncode == 0:
                result = json.loads(response.stdout)
                return result.get('response', '').strip()
        except Exception as e:
            print(f"Error: {e}")
        
        return ""
    
    def _get_gpu_memory(self):
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0


def main():
    """Run the complete pattern interpretation experiment"""
    print("=" * 70)
    print("PHI3 PATTERN INTERPRETATION STATE EXPERIMENT")
    print("=" * 70)
    
    experiment = Phi3PatternInterpretation()
    
    # Phase 1: Initial state
    print("\nPhase 1: Initial State Capture")
    initial_state = experiment.capture_initial_state()
    
    # Phase 2: First interpretation
    print("\nPhase 2: First Pattern Interpretation")
    first_interpretation = experiment.interpret_patterns("first")
    
    # Phase 3: Second interpretation (same text)
    print("\nPhase 3: Second Pattern Interpretation (same text)")
    time.sleep(5)  # Brief pause
    second_interpretation = experiment.interpret_patterns("second")
    
    # Phase 4: Final state
    print("\nPhase 4: Final State Capture")
    final_state = experiment.capture_final_state()
    
    # Phase 5: Analysis
    print("\nPhase 5: Analysis")
    
    # Compare states
    print("\n📊 State Comparison:")
    print(f"  Initial test response hash: {initial_state['test_response_hash'][:32]}...")
    print(f"  Final test response hash:   {final_state['test_response_hash'][:32]}...")
    print(f"  State changed: {'YES' if initial_state['test_response_hash'] != final_state['test_response_hash'] else 'NO'}")
    
    # Compare interpretations
    interpretation_comparison = experiment.compare_interpretations(
        first_interpretation, 
        second_interpretation
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"✓ Patterns.txt processed in {len(first_interpretation['chunk_interpretations'])} chunks")
    print(f"✓ Each chunk interpreted twice with creative temperature (0.7)")
    print(f"✓ Model state: {'CHANGED' if initial_state['test_response_hash'] != final_state['test_response_hash'] else 'UNCHANGED'}")
    print(f"✓ Interpretation consistency: {len(interpretation_comparison['differences'])} differences found")
    
    if len(interpretation_comparison['differences']) > 0:
        print("\n🔬 FINDING: Phi3 shows non-deterministic interpretation behavior")
        print("   Creative tasks (temp>0) produce varied outputs even with same input")
    else:
        print("\n🔬 FINDING: Phi3 maintains consistent interpretations")
    
    print(f"\nResults saved in: {experiment.results_dir}")


if __name__ == "__main__":
    main()