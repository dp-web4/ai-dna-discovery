#!/usr/bin/env python3
"""
Universal Patterns Test - SAFE VERSION
Tests one model at a time with resource monitoring
"""

import json
import urllib.request
import time
import os
import subprocess
from datetime import datetime
from distributed_memory import DistributedMemory
from claude_orchestrator_memory import ClaudeOrchestratorMemory

class SafeUniversalPatternsTest:
    def __init__(self):
        self.dm = DistributedMemory()
        self.orchestrator = ClaudeOrchestratorMemory()
        self.models = [
            'phi3:mini', 'tinyllama', 'gemma:2b',
            'mistral:latest', 'deepseek-coder:1.3b', 'qwen:0.5b'
        ]
        
        # Simplified test patterns
        self.test_patterns = {
            'existence': {'symbol': '∃', 'test': 'Does ∃ mean exists?'},
            'consciousness': {'symbol': 'Ψ', 'test': 'Does Ψ mean consciousness?'},
            'infinity': {'symbol': '∞', 'test': 'Does ∞ mean infinite?'}
        }
        
        # Safety thresholds
        self.memory_threshold_mb = 500  # Stop if free memory < 500MB
        self.timeout_seconds = 30  # Shorter timeout
        self.delay_between_models = 5  # Seconds between models
    
    def check_resources(self):
        """Check if system has enough resources"""
        try:
            # Use free command to check memory
            result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            # Parse memory line (second line)
            if len(lines) > 1:
                mem_parts = lines[1].split()
                if len(mem_parts) >= 7:
                    available_mb = int(mem_parts[6])  # Available column
                    print(f"💾 Available memory: {available_mb}MB")
                    
                    if available_mb < self.memory_threshold_mb:
                        print(f"⚠️  Low memory! Only {available_mb}MB available")
                        return False
                    return True
        except:
            # If we can't check, assume it's OK
            print("💾 Memory check unavailable, proceeding...")
            return True
    
    def query_model_safe(self, model, prompt):
        """Query with safety checks"""
        if not self.check_resources():
            return "SKIPPED: Low memory"
        
        url = 'http://localhost:11434/api/generate'
        data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_predict': 20  # Very short responses
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', '')
        except Exception as e:
            return f"Error: {str(e)}"
    
    def test_single_model(self, model_name):
        """Test just one model"""
        print(f"\n🤖 Testing {model_name}")
        print("=" * 40)
        
        results = {}
        
        # Warm up
        print("Warming up...")
        self.query_model_safe(model_name, "Hi")
        time.sleep(2)
        
        # Test patterns
        for pattern_name, pattern_info in self.test_patterns.items():
            print(f"\n📍 {pattern_name}: {pattern_info['symbol']}")
            
            start = time.time()
            response = self.query_model_safe(model_name, pattern_info['test'])
            elapsed = time.time() - start
            
            # Check understanding
            understood = False
            if isinstance(response, str) and not response.startswith(("Error", "SKIPPED")):
                affirmative = ['yes', 'correct', 'true', 'indeed']
                understood = any(word in response.lower() for word in affirmative)
            
            print(f"   Response: {response[:50]}...")
            print(f"   Understood: {'✓' if understood else '✗'}")
            print(f"   Time: {elapsed:.1f}s")
            
            results[pattern_name] = {
                'response': response,
                'understood': understood,
                'time': elapsed
            }
            
            # Store in distributed memory
            self.dm.add_memory(
                session_id=f"universal_patterns_safe_{model_name}",
                user_input=pattern_info['test'],
                ai_response=response,
                model=model_name,
                response_time=elapsed,
                facts={
                    'pattern': [(pattern_name, 1.0)],
                    'symbol': [(pattern_info['symbol'], 1.0)],
                    'understood': [('success', 1.0 if understood else 0.0)]
                }
            )
            
            # Save incrementally
            self.save_progress(model_name, results)
            
            # Delay between patterns
            time.sleep(2)
        
        return results
    
    def save_progress(self, model_name, results):
        """Save progress after each test"""
        filename = f'universal_patterns_safe_{model_name.replace(":", "_")}.json'
        with open(filename, 'w') as f:
            json.dump({
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        print(f"   💾 Saved to {filename}")
    
    def run_safe_test(self):
        """Run tests safely, one model at a time"""
        print("🛡️  SAFE UNIVERSAL PATTERNS TEST")
        print(f"Device: {self.dm.device_id}")
        print(f"Memory threshold: {self.memory_threshold_mb}MB")
        print(f"Timeout: {self.timeout_seconds}s")
        
        # Log test start with orchestrator
        self.orchestrator.track_test_execution(
            test_name="universal_patterns_safe",
            models=self.models,
            status="started",
            notes=f"Resource-limited test on {self.dm.device_id}"
        )
        
        all_results = {}
        
        for i, model in enumerate(self.models):
            print(f"\n\n{'='*60}")
            print(f"MODEL {i+1}/{len(self.models)}")
            
            if not self.check_resources():
                print(f"⚠️  Stopping - low memory before testing {model}")
                self.orchestrator.track_system_issue(
                    issue_type="low_memory",
                    description=f"Stopped before testing {model} - memory below {self.memory_threshold_mb}MB"
                )
                break
            
            results = self.test_single_model(model)
            all_results[model] = results
            
            # Track model performance
            successes = sum(1 for r in results.values() if r['understood'])
            avg_time = sum(r['time'] for r in results.values()) / len(results)
            
            self.orchestrator.track_model_performance(
                model=model,
                test="universal_patterns",
                metrics={
                    'success_rate': successes / len(results),
                    'avg_response_time': avg_time,
                    'patterns_tested': len(results)
                }
            )
            
            # Save overall progress
            with open('universal_patterns_safe_all.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            
            if i < len(self.models) - 1:
                print(f"\n⏳ Waiting {self.delay_between_models}s before next model...")
                time.sleep(self.delay_between_models)
        
        # Summary
        print("\n\n📊 SUMMARY")
        print("=" * 60)
        tested = len(all_results)
        print(f"Models tested: {tested}/{len(self.models)}")
        
        if tested > 0:
            # Success rates
            for model, results in all_results.items():
                successes = sum(1 for r in results.values() if r['understood'])
                rate = successes / len(results) * 100
                print(f"{model}: {successes}/{len(results)} ({rate:.0f}%)")
        
        # Show distributed memory summary
        self.show_memory_summary()
        
        # Log test completion
        self.orchestrator.track_test_execution(
            test_name="universal_patterns_safe",
            models=list(all_results.keys()),
            status="completed",
            notes=f"Tested {tested}/{len(self.models)} models successfully"
        )
        
        # Show orchestrator status
        self.orchestrator.show_summary()
    
    def show_memory_summary(self):
        """Show what's stored in distributed memory"""
        print("\n\n💾 DISTRIBUTED MEMORY STATE")
        print("=" * 60)
        
        # Query recent memories
        recent = self.dm.get_recent_memories(limit=10)
        
        if recent:
            print(f"Total memories stored: {len(recent)}")
            
            # Count by model
            model_counts = {}
            for memory in recent:
                model = memory.get('model', 'unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            
            print("\nMemories by model:")
            for model, count in sorted(model_counts.items()):
                if not model.startswith('claude:orchestrator'):
                    print(f"  {model}: {count} entries")
        else:
            print("No memories found in distributed system")

def main():
    """Run safe test"""
    tester = SafeUniversalPatternsTest()
    
    print("🚀 Starting SAFE universal patterns test")
    print("This version:")
    print("- Tests one model at a time")
    print("- Monitors memory usage")
    print("- Uses shorter timeouts")
    print("- Saves progress after each test")
    
    tester.run_safe_test()
    
    print("\n✅ Test completed safely!")

if __name__ == "__main__":
    main()