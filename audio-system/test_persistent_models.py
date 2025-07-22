#!/usr/bin/env python3
"""
Test keeping multiple models persistent in GPU memory
Goal: Prevent model swapping overhead
"""

import time
import subprocess
import json
import requests
import threading
from typing import Dict, List

class PersistentModelManager:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.models = ["phi3:mini", "gemma:2b", "tinyllama:latest"]
        self.active_models = {}
        
    def check_ollama_status(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def list_loaded_models(self):
        """List currently loaded models in memory"""
        try:
            response = requests.get(f"{self.base_url}/api/ps")
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def load_model_persistent(self, model: str):
        """Load a model and keep it warm in memory"""
        print(f"\nLoading {model} into GPU memory...")
        
        # Method 1: Use keep_alive parameter
        data = {
            "model": model,
            "prompt": "Hello",
            "keep_alive": "24h"  # Keep model in memory for 24 hours
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                stream=True
            )
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    # Just consume the response to complete the load
                    pass
                    
            print(f"✓ {model} loaded with 24h keep_alive")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {model}: {e}")
            return False
    
    def keep_models_warm(self, models: List[str], interval: int = 300):
        """Periodically ping models to keep them in memory"""
        def warm_model(model):
            data = {
                "model": model,
                "prompt": "",  # Empty prompt just to keep warm
                "keep_alive": "24h"
            }
            try:
                requests.post(f"{self.base_url}/api/generate", json=data)
            except:
                pass
        
        print(f"\nStarting keep-warm thread (interval: {interval}s)...")
        
        def warm_loop():
            while True:
                for model in models:
                    warm_model(model)
                time.sleep(interval)
        
        thread = threading.Thread(target=warm_loop, daemon=True)
        thread.start()
    
    def test_model_persistence(self):
        """Test different methods of keeping models persistent"""
        
        print("=== Testing Model Persistence Methods ===\n")
        
        # Check Ollama status
        if not self.check_ollama_status():
            print("❌ Ollama server not running!")
            return
        
        # Show current state
        print("Current loaded models:")
        loaded = self.list_loaded_models()
        if loaded:
            for model in loaded.get('models', []):
                print(f"  - {model['name']} (size: {model.get('size', 'unknown')})")
        else:
            print("  None")
        
        # Method 1: Load with keep_alive
        print("\n### Method 1: Using keep_alive parameter ###")
        for model in self.models[:2]:  # Load first 2 models
            self.load_model_persistent(model)
            time.sleep(2)
        
        # Check GPU memory usage
        print("\nChecking GPU memory after loading...")
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
        
        # Show loaded models
        print("\nCurrently loaded models:")
        loaded = self.list_loaded_models()
        for model in loaded.get('models', []):
            print(f"  - {model['name']} (expires: {model.get('expires_at', 'unknown')})")
        
        # Method 2: Start keep-warm thread
        print("\n### Method 2: Keep-warm thread ###")
        self.keep_models_warm(self.models[:2], interval=60)
        
        # Test rapid switching
        print("\n### Testing rapid model switching ###")
        print("Without persistence, this would cause GPU swapping...")
        
        start_time = time.time()
        for i in range(6):
            model = self.models[i % 2]
            data = {
                "model": model,
                "prompt": "Count to 3",
                "keep_alive": "24h",
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=data)
            print(f"  Request {i+1} to {model}: {response.status_code}")
        
        elapsed = time.time() - start_time
        print(f"\nCompleted 6 requests in {elapsed:.2f}s")
        print("(Should be fast since models stay in memory)")
        
        # Show final state
        print("\nFinal GPU state:")
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
        
        print("\nFinal loaded models:")
        loaded = self.list_loaded_models()
        for model in loaded.get('models', []):
            print(f"  - {model['name']}")
    
    def configure_ollama_server(self):
        """Show how to configure Ollama for better persistence"""
        
        print("\n=== Ollama Server Configuration ===")
        print("\nTo improve model persistence, set these environment variables:")
        print("export OLLAMA_KEEP_ALIVE=24h  # Default keep-alive time")
        print("export OLLAMA_MAX_LOADED_MODELS=4  # Allow more models in memory")
        print("export OLLAMA_NUM_PARALLEL=4  # Handle parallel requests")
        print("\nThen restart Ollama:")
        print("systemctl restart ollama  # or ollama serve")
        
        # Check current settings
        print("\nChecking current Ollama environment...")
        env_vars = ["OLLAMA_KEEP_ALIVE", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL"]
        for var in env_vars:
            value = subprocess.run(["printenv", var], capture_output=True, text=True).stdout.strip()
            print(f"{var}: {value if value else '(not set)'}")

if __name__ == "__main__":
    manager = PersistentModelManager()
    
    # Show configuration options
    manager.configure_ollama_server()
    
    # Test persistence methods
    manager.test_model_persistence()
    
    print("\n💡 Tips for persistent models:")
    print("1. Use 'keep_alive' parameter in all requests")
    print("2. Set OLLAMA_KEEP_ALIVE environment variable")
    print("3. Run a keep-warm thread for critical models")
    print("4. Increase OLLAMA_MAX_LOADED_MODELS if needed")