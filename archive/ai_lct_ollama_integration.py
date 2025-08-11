#!/usr/bin/env python3
"""
AI LCT Ollama Integration - Real interaction tracking with local AI models
Integrates LCT tracking with actual Ollama API calls
"""

import json
import time
import requests
from typing import Dict, Optional
from ai_lct_experiment import AIEntityManager, LCT


class OllamaLCTClient:
    """Client that wraps Ollama API with LCT tracking"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.manager = AIEntityManager()
        self.active_lcts: Dict[str, LCT] = {}
    
    def register_model(self, model_name: str) -> LCT:
        """Register an Ollama model with an LCT"""
        if model_name not in self.active_lcts:
            lct = self.manager.create_ai_lct(model_name, self.base_url)
            self.active_lcts[model_name] = lct
            print(f"Registered {model_name} with LCT: {lct.entity_id}")
        return self.active_lcts[model_name]
    
    def generate(self, model: str, prompt: str, energy_cost: float = 5.0) -> Dict:
        """Generate text with LCT tracking"""
        
        # Ensure model is registered
        lct = self.register_model(model)
        
        # Check energy availability
        if lct.atp_balance < energy_cost:
            return {
                "error": f"Insufficient energy. ATP balance: {lct.atp_balance}, required: {energy_cost}"
            }
        
        start_time = time.time()
        
        try:
            # Make actual API call to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Calculate quality metrics
            response_time = time.time() - start_time
            response_text = result.get("response", "")
            
            # Simple quality heuristics
            quality_score = min(1.0, len(response_text) / 500)  # Longer responses score higher
            if response_time < 2.0:
                quality_score *= 1.2  # Bonus for fast response
            quality_score = min(1.0, quality_score)
            
            # Track successful interaction
            interaction_result = {
                "success": True,
                "quality_score": quality_score,
                "response_time": response_time,
                "response_length": len(response_text),
                "model": model
            }
            
            self.manager.track_interaction(
                lct,
                "text_generation",
                energy_cost,
                interaction_result
            )
            
            return {
                "response": response_text,
                "lct_id": lct.entity_id,
                "atp_remaining": lct.atp_balance,
                "quality_score": quality_score,
                "response_time": response_time
            }
            
        except Exception as e:
            # Track failed interaction
            self.manager.track_interaction(
                lct,
                "text_generation",
                energy_cost / 2,  # Partial energy cost for failed attempt
                {"success": False, "error": str(e)}
            )
            
            return {"error": str(e)}
    
    def create_task_lct(self, task_description: str) -> LCT:
        """Create an LCT for a specific task"""
        entity_id = f"task_{int(time.time())}"
        
        task_lct = LCT(
            entity_id=entity_id,
            entity_type="task",
            entity_subtype="ai_interaction",
        )
        
        # Tasks don't have trust/value tensors initially
        self.manager.registry[entity_id] = task_lct
        self.manager.save_registry()
        
        return task_lct
    
    def link_task_to_ai(self, task_lct: LCT, ai_model: str):
        """Link a task LCT to an AI model LCT"""
        ai_lct = self.register_model(ai_model)
        
        task_lct.add_link("performed_by", ai_lct.entity_id)
        ai_lct.add_link("performed_task", task_lct.entity_id)
        
        self.manager.save_registry()
    
    def get_model_stats(self, model: str) -> Dict:
        """Get statistics for a model's LCT"""
        lct = self.active_lcts.get(model)
        if not lct:
            return {"error": "Model not registered"}
        
        return {
            "entity_id": lct.entity_id,
            "model": model,
            "atp_balance": lct.atp_balance,
            "adp_balance": lct.adp_balance,
            "trust_score": lct.t3.aggregate_score() if lct.t3 else 0,
            "value_score": lct.v3.aggregate_score() if lct.v3 else 0,
            "total_interactions": len(lct.interactions),
            "successful_interactions": sum(
                1 for i in lct.interactions 
                if i.get("result", {}).get("success", False)
            )
        }


def demo_ollama_lct_integration():
    """Demonstrate the Ollama LCT integration"""
    
    client = OllamaLCTClient()
    
    print("=== Ollama LCT Integration Demo ===\n")
    
    # Create a task LCT
    task = client.create_task_lct("Explain the concept of trust in distributed systems")
    print(f"Created task LCT: {task.entity_id}")
    
    # Generate response with Phi-3
    print("\nGenerating response with Phi-3...")
    result = client.generate(
        "phi3:mini",
        "Explain the concept of trust in distributed systems in 2-3 sentences.",
        energy_cost=10.0
    )
    
    if "error" not in result:
        print(f"\nResponse: {result['response'][:200]}...")
        print(f"Quality score: {result['quality_score']:.3f}")
        print(f"Response time: {result['response_time']:.2f}s")
        print(f"ATP remaining: {result['atp_remaining']}")
        
        # Link task to AI
        client.link_task_to_ai(task, "phi3:mini")
    
    # Check model stats
    stats = client.get_model_stats("phi3:mini")
    print(f"\nModel Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Simulate value creation and attestation
    if stats["successful_interactions"] > 0:
        # Create a validator (could be another AI or human reviewer)
        validator = client.manager.create_ai_lct("human_validator")
        validator.t3.talent.value = 0.95
        validator.t3.talent.confidence = 1.0
        
        # Get the AI's LCT
        ai_lct = client.active_lcts["phi3:mini"]
        
        # Create value attestation
        attestation = client.manager.create_value_attestation(
            ai_lct,
            validator,
            value_score=0.9,
            description="High-quality explanation of distributed trust concepts"
        )
        
        print(f"\nValue Attestation Created:")
        print(f"Energy recharged: {attestation['energy_recharged']:.2f} ATP")
        
        # Check updated stats
        stats = client.get_model_stats("phi3:mini")
        print(f"\nUpdated Model Statistics:")
        print(f"ATP balance: {stats['atp_balance']:.2f}")
        print(f"Trust score: {stats['trust_score']:.3f}")
        print(f"Value score: {stats['value_score']:.3f}")


def monitor_ai_ecosystem():
    """Monitor multiple AI models in the ecosystem"""
    
    client = OllamaLCTClient()
    
    # List available models
    try:
        response = requests.get(f"{client.base_url}/api/tags")
        models = [m["name"] for m in response.json().get("models", [])]
        
        print("Available models:", models)
        
        # Register all models
        for model in models[:3]:  # Limit to first 3 for demo
            client.register_model(model)
        
        # Run a task across multiple models
        prompt = "What is the meaning of consciousness?"
        
        for model in models[:3]:
            print(f"\n--- Testing {model} ---")
            result = client.generate(model, prompt, energy_cost=15.0)
            
            if "error" not in result:
                print(f"Quality: {result['quality_score']:.3f}")
                print(f"Time: {result['response_time']:.2f}s")
                print(f"ATP remaining: {result['atp_remaining']}")
    
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")


if __name__ == "__main__":
    # Run the demo
    demo_ollama_lct_integration()
    
    print("\n" + "="*50 + "\n")
    
    # Monitor ecosystem
    monitor_ai_ecosystem()