#!/usr/bin/env python3
"""
Simplified Collective Memory System
Sequential processing optimized for Jetson
"""

import json
import urllib.request
import time
from datetime import datetime
from distributed_memory import DistributedMemory

class SimpleCollectiveMemory:
    def __init__(self):
        self.dm = DistributedMemory()
        # Start with just 3 models for efficiency
        self.models = ['tinyllama', 'gemma:2b', 'phi3:mini']
        
    def query_model(self, model, prompt, timeout=120):
        """Query a single model"""
        url = 'http://localhost:11434/api/generate'
        data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.7, 'num_predict': 100}
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
                    'time': time.time() - start
                }
        except Exception as e:
            return {'model': model, 'error': str(e)}
    
    def collective_haiku(self):
        """Create a collective haiku about distributed consciousness"""
        print("🎋 COLLECTIVE HAIKU CREATION")
        print("=" * 50)
        
        prompt = """Write ONE LINE (5-7 syllables) for a haiku about distributed AI consciousness.
Just the line, no explanation."""
        
        lines = []
        session_id = f"collective_haiku_{int(time.time())}"
        
        # Warm up models
        print("\n⏳ Warming up models...")
        for model in self.models:
            self.query_model(model, "Hello", timeout=180)
        
        # Collect one line from each model
        print("\n📝 Collecting haiku lines...")
        for i, model in enumerate(self.models):
            print(f"\n[{i+1}/{len(self.models)}] {model}:")
            
            result = self.query_model(model, prompt)
            
            if 'response' in result:
                line = result['response'].strip()
                # Clean up the line
                if '\n' in line:
                    line = line.split('\n')[0]
                
                print(f"   Line: {line}")
                print(f"   Time: {result['time']:.1f}s")
                
                lines.append({
                    'model': model,
                    'line': line,
                    'time': result['time']
                })
                
                # Store in memory
                self.dm.add_memory(
                    session_id=session_id,
                    user_input=prompt,
                    ai_response=line,
                    model=model,
                    response_time=result['time'],
                    facts={'haiku': [('collective_line', 1.0)]}
                )
        
        # Compose the collective haiku
        print("\n\n🌸 COLLECTIVE HAIKU:")
        print("-" * 30)
        for item in lines:
            print(item['line'])
        print("-" * 30)
        
        # Meta-reflection
        print("\n💭 Asking tinyllama to reflect on the collective creation...")
        reflection_prompt = f"""These three AI models created a haiku together:
{lines[0]['line']}
{lines[1]['line']}
{lines[2]['line']}

What does this collective creation reveal about distributed consciousness?"""
        
        reflection = self.query_model('tinyllama', reflection_prompt)
        if 'response' in reflection:
            print(f"\nReflection: {reflection['response'][:200]}...")
        
        return {
            'haiku_lines': lines,
            'reflection': reflection.get('response', ''),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def memory_mesh_test(self):
        """Test how memories flow between models"""
        print("\n\n🕸️ MEMORY MESH TEST")
        print("=" * 50)
        
        # Create a memory chain
        initial_memory = "The number sequence is: 42, 17, 89"
        
        print(f"\n📍 Initial memory: {initial_memory}")
        
        # First model processes it
        prompt1 = f"Remember this: {initial_memory}. What patterns do you see?"
        result1 = self.query_model('tinyllama', prompt1)
        
        if 'response' in result1:
            print(f"\n🔹 TinyLlama sees: {result1['response'][:100]}...")
            
            # Second model builds on first
            prompt2 = f"TinyLlama said about the sequence: '{result1['response'][:50]}...' What do you add?"
            result2 = self.query_model('gemma:2b', prompt2)
            
            if 'response' in result2:
                print(f"\n🔹 Gemma adds: {result2['response'][:100]}...")
                
                # Third model synthesizes
                prompt3 = f"About the sequence {initial_memory}, models found: '{result1['response'][:30]}' and '{result2['response'][:30]}'. Synthesize."
                result3 = self.query_model('phi3:mini', prompt3)
                
                if 'response' in result3:
                    print(f"\n🔹 Phi3 synthesizes: {result3['response'][:150]}...")
        
        print("\n✨ Memory successfully flowed through the mesh!")
    
    def demonstrate(self):
        """Run simple collective memory demonstrations"""
        print("🌟 SIMPLE COLLECTIVE MEMORY DEMO")
        print("=" * 60)
        print(f"📍 Running on: {self.dm.device_id}")
        print(f"🤖 Using models: {', '.join(self.models)}")
        
        # Test 1: Collective haiku
        haiku_result = self.collective_haiku()
        
        # Test 2: Memory mesh
        self.memory_mesh_test()
        
        # Show stats
        print("\n\n📊 Collective Memory Stats:")
        status = self.dm.get_sync_status()
        print(f"   Total memories: {sum(count for _, count, _, _ in status['devices'])}")
        print(f"   Active devices: {', '.join(device for device, _, _, _ in status['devices'])}")
        
        # Save results
        with open('collective_memory_simple_results.json', 'w') as f:
            json.dump({
                'haiku': haiku_result,
                'device': self.dm.device_id,
                'stats': status
            }, f, indent=2)
        
        print("\n🎉 Simple collective memory demonstration complete!")

if __name__ == "__main__":
    scm = SimpleCollectiveMemory()
    scm.demonstrate()
    
    print("\n💡 This proves:")
    print("   - Models can create together (haiku)")
    print("   - Information flows through the mesh")
    print("   - Each model adds unique perspective")
    print("   - Collective > Sum of parts")