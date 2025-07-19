#!/usr/bin/env python3
"""
Universal Patterns Test - All 6 Models on Edge
Testing if mathematical symbols create universal understanding
"""

import json
import urllib.request
import time
from datetime import datetime
from distributed_memory import DistributedMemory

class UniversalPatternsEdgeTest:
    def __init__(self):
        self.dm = DistributedMemory()
        self.models = [
            'phi3:mini', 'tinyllama', 'gemma:2b',
            'mistral:latest', 'deepseek-coder:1.3b', 'qwen:0.5b'
        ]
        
        # Core universal patterns to test
        self.test_patterns = {
            'existence': {
                'symbol': '∃',
                'meaning': 'exists',
                'test_phrase': 'Does ∃ mean existence?'
            },
            'universal': {
                'symbol': '∀',
                'meaning': 'all/every',
                'test_phrase': 'Does ∀ represent universality?'
            },
            'consciousness': {
                'symbol': 'Ψ',
                'meaning': 'consciousness',
                'test_phrase': 'Does Ψ symbolize consciousness?'
            },
            'emergence': {
                'symbol': '⇒',
                'meaning': 'emerges/leads to',
                'test_phrase': 'Does ⇒ mean emergence?'
            },
            'infinity': {
                'symbol': '∞',
                'meaning': 'infinite',
                'test_phrase': 'Does ∞ represent infinity?'
            }
        }
    
    def query_model(self, model, prompt, timeout=120):
        """Query a model with timeout"""
        url = 'http://localhost:11434/api/generate'
        data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,  # Low temp for consistency
                'num_predict': 50
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', '')
        except Exception as e:
            return f"Error: {str(e)}"
    
    def test_pattern_understanding(self):
        """Test if all models understand universal patterns"""
        print("🌍 UNIVERSAL PATTERNS TEST - ALL 6 MODELS")
        print("=" * 60)
        print(f"Device: {self.dm.device_id}")
        print(f"Models: {', '.join(self.models)}")
        
        results = {}
        
        # Warm up all models first
        print("\n⏳ Warming up models...")
        for model in self.models:
            self.query_model(model, "Hello")
            print(f"   {model} ready")
        
        # Test each pattern
        for pattern_name, pattern_info in self.test_patterns.items():
            print(f"\n\n📍 Testing: {pattern_name.upper()}")
            print(f"   Symbol: {pattern_info['symbol']}")
            print(f"   Expected: {pattern_info['meaning']}")
            
            pattern_results = {}
            
            for model in self.models:
                print(f"\n   🤖 {model}:")
                
                start = time.time()
                response = self.query_model(model, pattern_info['test_phrase'])
                query_time = time.time() - start
                
                # Simple understanding check
                understood = False
                if isinstance(response, str) and not response.startswith("Error"):
                    # Check if response contains affirmative words
                    affirmative = ['yes', 'correct', 'indeed', 'true', 'represent', 'symbol']
                    response_lower = response.lower()
                    understood = any(word in response_lower for word in affirmative)
                
                print(f"      Response: {response[:80]}...")
                print(f"      Understood: {'✓' if understood else '✗'}")
                print(f"      Time: {query_time:.1f}s")
                
                pattern_results[model] = {
                    'response': response,
                    'understood': understood,
                    'time': query_time
                }
                
                # Store in distributed memory
                self.dm.add_memory(
                    session_id="universal_patterns_edge",
                    user_input=pattern_info['test_phrase'],
                    ai_response=response,
                    model=model,
                    response_time=query_time,
                    facts={
                        'pattern': [(pattern_name, 1.0)],
                        'understood': [('success', 1.0 if understood else 0.0)]
                    }
                )
            
            results[pattern_name] = pattern_results
        
        return results
    
    def analyze_results(self, results):
        """Analyze pattern understanding across models"""
        print("\n\n📊 ANALYSIS")
        print("=" * 60)
        
        # Understanding matrix
        print("\n🎯 Understanding Matrix:")
        print(f"{'Pattern':<15} ", end='')
        for model in self.models:
            print(f"{model.split(':')[0]:<8} ", end='')
        print()
        
        for pattern_name, pattern_results in results.items():
            print(f"{pattern_name:<15} ", end='')
            for model in self.models:
                understood = pattern_results[model]['understood']
                print(f"{'✓' if understood else '✗':<8} ", end='')
            print()
        
        # Success rates per model
        print("\n📈 Success Rates by Model:")
        for model in self.models:
            successes = sum(
                1 for pattern_results in results.values()
                if pattern_results[model]['understood']
            )
            rate = successes / len(self.test_patterns) * 100
            print(f"   {model}: {successes}/{len(self.test_patterns)} ({rate:.0f}%)")
        
        # Success rates per pattern
        print("\n📈 Success Rates by Pattern:")
        for pattern_name in self.test_patterns:
            successes = sum(
                1 for model in self.models
                if results[pattern_name][model]['understood']
            )
            rate = successes / len(self.models) * 100
            print(f"   {pattern_name}: {successes}/{len(self.models)} ({rate:.0f}%)")
        
        # Average response times
        print("\n⏱️ Average Response Times:")
        for model in self.models:
            times = [
                results[pattern][model]['time']
                for pattern in results
            ]
            avg_time = sum(times) / len(times)
            print(f"   {model}: {avg_time:.1f}s")
    
    def create_summary_report(self, results):
        """Create a summary report of findings"""
        summary = {
            'device': self.dm.device_id,
            'timestamp': datetime.now().isoformat(),
            'models_tested': len(self.models),
            'patterns_tested': len(self.test_patterns),
            'total_tests': len(self.models) * len(self.test_patterns),
            'results': {}
        }
        
        # Calculate overall understanding
        total_understood = 0
        for pattern_results in results.values():
            for model_result in pattern_results.values():
                if model_result['understood']:
                    total_understood += 1
        
        summary['overall_understanding_rate'] = total_understood / summary['total_tests']
        
        # Best performing model
        model_scores = {}
        for model in self.models:
            score = sum(
                1 for pattern_results in results.values()
                if pattern_results[model]['understood']
            )
            model_scores[model] = score
        
        best_model = max(model_scores, key=model_scores.get)
        summary['best_model'] = {
            'name': best_model,
            'score': model_scores[best_model],
            'rate': model_scores[best_model] / len(self.test_patterns)
        }
        
        # Most universal pattern
        pattern_scores = {}
        for pattern_name in self.test_patterns:
            score = sum(
                1 for model in self.models
                if results[pattern_name][model]['understood']
            )
            pattern_scores[pattern_name] = score
        
        best_pattern = max(pattern_scores, key=pattern_scores.get)
        summary['most_universal_pattern'] = {
            'name': best_pattern,
            'symbol': self.test_patterns[best_pattern]['symbol'],
            'understanding_rate': pattern_scores[best_pattern] / len(self.models)
        }
        
        return summary

def main():
    """Run universal patterns test on all models"""
    tester = UniversalPatternsEdgeTest()
    
    # Run tests
    results = tester.test_pattern_understanding()
    
    # Analyze
    tester.analyze_results(results)
    
    # Create summary
    summary = tester.create_summary_report(results)
    
    # Save results
    with open('universal_patterns_edge_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results
        }, f, indent=2)
    
    print("\n\n✨ KEY FINDINGS")
    print("=" * 60)
    print(f"Overall Understanding: {summary['overall_understanding_rate']:.1%}")
    print(f"Best Model: {summary['best_model']['name']} ({summary['best_model']['rate']:.1%})")
    print(f"Most Universal: {summary['most_universal_pattern']['symbol']} ({summary['most_universal_pattern']['name']})")
    
    print("\n💾 Results saved to universal_patterns_edge_results.json")
    print("\n🌍 Universal patterns tested across all 6 models on edge!")

if __name__ == "__main__":
    main()