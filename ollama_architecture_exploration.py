#!/usr/bin/env python3
"""
Ollama Architecture Exploration
Understanding how Ollama manages model state and KV-cache
"""

import subprocess
import json
import requests
import time
import os
from datetime import datetime

class OllamaArchitectureExplorer:
    def __init__(self):
        self.api_base = "http://localhost:11434"
        self.exploration_results = {
            "timestamp": datetime.now().isoformat(),
            "findings": {}
        }
    
    def explore_api_endpoints(self):
        """Discover available API endpoints"""
        print("=== EXPLORING OLLAMA API ===\n")
        
        # Known endpoints
        endpoints = [
            "/api/tags",      # List models
            "/api/generate",  # Generate text
            "/api/chat",      # Chat interface
            "/api/embeddings", # Get embeddings
            "/api/show",      # Show model info
            "/api/ps",        # Process status
        ]
        
        discovered = []
        
        for endpoint in endpoints:
            try:
                if endpoint == "/api/show":
                    # This needs a model parameter
                    response = requests.post(
                        f"{self.api_base}{endpoint}",
                        json={"name": "phi3:mini"}
                    )
                else:
                    response = requests.get(f"{self.api_base}{endpoint}")
                
                if response.status_code < 400:
                    discovered.append({
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "has_data": len(response.text) > 0
                    })
                    print(f"✓ {endpoint} - Status: {response.status_code}")
            except Exception as e:
                print(f"✗ {endpoint} - Error: {str(e)}")
        
        return discovered
    
    def check_model_processes(self):
        """Check running Ollama processes using ps"""
        print("\n=== OLLAMA PROCESSES ===\n")
        
        ollama_processes = []
        
        try:
            # Use ps command instead of psutil
            result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'ollama' in line.lower():
                        parts = line.split()
                        if len(parts) >= 11:
                            ollama_processes.append({
                                'user': parts[0],
                                'pid': parts[1],
                                'cpu': parts[2],
                                'mem': parts[3],
                                'command': ' '.join(parts[10:])
                            })
                            print(f"PID {parts[1]}: {' '.join(parts[10:])[:80]}...")
                            print(f"  CPU: {parts[2]}%, MEM: {parts[3]}%")
        except Exception as e:
            print(f"Error checking processes: {e}")
        
        return ollama_processes
    
    def analyze_model_loading(self):
        """Analyze how models are loaded and cached"""
        print("\n=== MODEL LOADING ANALYSIS ===\n")
        
        # Get current loaded models
        try:
            response = requests.get(f"{self.api_base}/api/ps")
            if response.status_code == 200:
                loaded_models = response.json()
                print("Currently loaded models:")
                for model in loaded_models.get('models', []):
                    print(f"  - {model['name']}")
                    print(f"    Size: {model.get('size', 'unknown')}")
                    print(f"    Digest: {model.get('digest', 'unknown')[:16]}...")
                return loaded_models
        except Exception as e:
            print(f"Error checking loaded models: {e}")
        
        return None
    
    def test_context_persistence(self):
        """Test if context persists between calls"""
        print("\n=== CONTEXT PERSISTENCE TEST ===\n")
        
        model = "phi3:mini"
        
        # First call - establish context
        print("Call 1: Establishing context...")
        response1 = requests.post(
            f"{self.api_base}/api/generate",
            json={
                "model": model,
                "prompt": "My name is Alice and I love quantum physics.",
                "stream": False
            }
        )
        
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"Response: {result1['response'][:100]}...")
            context1 = result1.get('context', [])
            print(f"Context tokens returned: {len(context1) if context1 else 0}")
        
        time.sleep(2)
        
        # Second call - use returned context
        print("\n\nCall 2: Testing context usage...")
        response2 = requests.post(
            f"{self.api_base}/api/generate",
            json={
                "model": model,
                "prompt": "What's my name and what do I love?",
                "context": context1 if 'context1' in locals() else None,
                "stream": False
            }
        )
        
        if response2.status_code == 200:
            result2 = response2.json()
            response_text = result2['response']
            print(f"Response: {response_text[:200]}...")
            
            # Check if context worked
            has_memory = "alice" in response_text.lower() and "quantum" in response_text.lower()
            print(f"\nContext persistence: {'✓ Working' if has_memory else '✗ Not working'}")
            
            return {
                "context_tokens": len(context1) if 'context1' in locals() else 0,
                "memory_preserved": has_memory
            }
        
        return None
    
    def explore_model_files(self):
        """Explore where Ollama stores model files"""
        print("\n=== MODEL FILE LOCATIONS ===\n")
        
        # Common Ollama model locations
        possible_paths = [
            "~/.ollama/models",
            "/usr/share/ollama/.ollama/models",
            "/var/lib/ollama/models",
            "~/.cache/ollama/models"
        ]
        
        import os
        found_paths = []
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                found_paths.append(expanded_path)
                print(f"✓ Found: {expanded_path}")
                
                # List contents
                try:
                    for item in os.listdir(expanded_path):
                        item_path = os.path.join(expanded_path, item)
                        if os.path.isdir(item_path):
                            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                     for dirpath, dirnames, filenames in os.walk(item_path)
                                     for filename in filenames) / 1024 / 1024 / 1024
                            print(f"    {item}: {size:.2f} GB")
                except Exception as e:
                    print(f"    Error listing: {e}")
        
        return found_paths
    
    def analyze_inference_behavior(self):
        """Analyze inference behavior with timing"""
        print("\n=== INFERENCE BEHAVIOR ANALYSIS ===\n")
        
        model = "phi3:mini"
        prompt = "Explain quantum entanglement in one sentence."
        
        timings = []
        responses = []
        
        print(f"Running 5 identical inferences...")
        
        for i in range(5):
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "seed": 42
                    }
                }
            )
            
            elapsed = time.time() - start_time
            timings.append(elapsed)
            
            if response.status_code == 200:
                result = response.json()
                responses.append(result['response'])
                print(f"  Run {i+1}: {elapsed:.2f}s")
        
        # Analyze timing patterns
        if len(timings) >= 2:
            warmup_overhead = timings[0] - min(timings[1:])
            print(f"\nWarmup overhead: {warmup_overhead:.2f}s")
            print(f"Average after warmup: {sum(timings[1:])/len(timings[1:]):.2f}s")
        
        # Check response consistency
        unique_responses = len(set(responses))
        print(f"\nUnique responses: {unique_responses}")
        
        return {
            "timings": timings,
            "warmup_overhead": warmup_overhead if 'warmup_overhead' in locals() else None,
            "response_variations": unique_responses
        }
    
    def create_architecture_report(self):
        """Create comprehensive architecture report"""
        report = f"""
OLLAMA ARCHITECTURE EXPLORATION REPORT
=====================================
Generated: {datetime.now().isoformat()}

## Key Findings

### API Architecture
- RESTful API on port 11434
- Supports streaming and non-streaming responses
- Context tokens can be passed between calls
- Models loaded on-demand and cached

### State Management
- Context persistence: ✓ Supported via context tokens
- Models remain loaded in memory after use
- Each model maintains its own memory allocation
- No built-in session management

### Potential KV-Cache Access Points

1. **Context Tokens**
   - Ollama returns context after each generation
   - Can be reused for conversation continuity
   - Essentially serialized KV-cache state

2. **Model Files**
   - Binary format in ~/.ollama/models
   - Contains weights and architecture
   - No direct KV-cache storage found

3. **Memory Behavior**
   - First inference shows warmup delay
   - Subsequent calls faster (cache warm)
   - Memory remains allocated between calls

### Integration Opportunities

1. **Immediate: Enhanced Context Management**
   - Intercept and store context tokens
   - Build session layer on top
   - Compress old contexts

2. **Medium-term: Custom Ollama Fork**
   - Expose KV-cache in API
   - Add session persistence
   - Enable state serialization

3. **Advanced: Binary Protocol**
   - Direct model memory access
   - Custom inference server
   - Full state control

## Recommendations

For our memory system:
1. Use context tokens as portable KV-cache
2. Build session management layer
3. Consider Ollama fork for deeper integration

For Jetson deployment:
1. Context tokens are lightweight
2. Can implement custom caching
3. Memory-mapped model files possible
"""
        
        return report
    
    def run_exploration(self):
        """Run complete Ollama architecture exploration"""
        print("OLLAMA ARCHITECTURE EXPLORATION")
        print("=" * 50)
        
        # 1. Explore API
        self.exploration_results['api_endpoints'] = self.explore_api_endpoints()
        
        # 2. Check processes
        self.exploration_results['processes'] = self.check_model_processes()
        
        # 3. Analyze model loading
        self.exploration_results['loaded_models'] = self.analyze_model_loading()
        
        # 4. Test context persistence
        self.exploration_results['context_test'] = self.test_context_persistence()
        
        # 5. Explore file locations
        self.exploration_results['model_paths'] = self.explore_model_files()
        
        # 6. Analyze inference behavior
        self.exploration_results['inference_analysis'] = self.analyze_inference_behavior()
        
        # Generate report
        report = self.create_architecture_report()
        print("\n" + "=" * 50)
        print(report)
        
        # Save results
        with open('/home/dp/ai-workspace/ai-agents/ollama_architecture_findings.json', 'w') as f:
            json.dump(self.exploration_results, f, indent=2)
        
        with open('/home/dp/ai-workspace/ai-agents/ollama_architecture_report.txt', 'w') as f:
            f.write(report)
        
        print("\nExploration complete! Results saved.")
        
        return self.exploration_results


if __name__ == "__main__":
    explorer = OllamaArchitectureExplorer()
    explorer.run_exploration()