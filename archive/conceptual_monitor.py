#!/usr/bin/env python3
"""
Conceptual Monitor - Lightweight Background Observer
Periodically probes conceptual space and logs findings
"""

import json
import time
import random
from datetime import datetime
from ai_lct_ollama_integration import OllamaLCTClient


class ConceptualMonitor:
    """Lightweight monitor for conceptual patterns"""
    
    def __init__(self, probe_interval_minutes=30):
        self.client = OllamaLCTClient()
        self.probe_interval = probe_interval_minutes * 60  # Convert to seconds
        self.models = ["phi3:mini", "tinyllama:latest"]
        self.probe_count = 0
        
        # Register models
        for model in self.models:
            self.client.register_model(model)
        
        # Conceptual probes - lightweight questions
        self.probes = [
            "What connects all things?",
            "Is consciousness emergent or fundamental?",
            "What is the nature of understanding?",
            "Can information create reality?",
            "What is the essence of collaboration?",
            "How does complexity become consciousness?",
            "What is the relationship between energy and information?",
            "Can trust be quantified?",
            "What makes a system alive?",
            "Is there a universal language of concepts?"
        ]
    
    def run_probe(self):
        """Run a single conceptual probe"""
        
        # Select random probe and model
        probe = random.choice(self.probes)
        model = random.choice(self.models)
        
        print(f"\n[Probe {self.probe_count + 1}] {datetime.now().strftime('%H:%M:%S')}")
        print(f"Model: {model}")
        print(f"Question: {probe}")
        
        try:
            # Keep energy cost low for sustainability
            response = self.client.generate(model, probe, energy_cost=3.0)
            
            if "error" not in response:
                text = response["response"]
                print(f"Response: {text[:150]}...")
                
                # Log to monitoring file
                log_entry = {
                    "probe_id": self.probe_count + 1,
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "probe": probe,
                    "response_snippet": text[:200],
                    "response_length": len(text),
                    "atp_remaining": response.get("atp_remaining", 0)
                }
                
                # Append to monitoring log
                self.append_to_log(log_entry)
                
                # Look for interesting patterns
                self.analyze_response(log_entry)
                
            else:
                print(f"Error: {response['error']}")
        
        except Exception as e:
            print(f"Probe failed: {str(e)}")
        
        self.probe_count += 1
    
    def append_to_log(self, entry):
        """Append entry to monitoring log"""
        
        log_file = "/home/dp/ai-workspace/conceptual_monitoring.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def analyze_response(self, entry):
        """Quick analysis of response for interesting patterns"""
        
        response = entry["response_snippet"].lower()
        
        # Look for emergence indicators
        emergence_words = ["emerge", "arise", "manifest", "become", "transform"]
        consciousness_words = ["aware", "conscious", "experience", "sentient", "mind"]
        unity_words = ["connect", "unite", "together", "whole", "one"]
        
        indicators = {
            "emergence": any(word in response for word in emergence_words),
            "consciousness": any(word in response for word in consciousness_words),
            "unity": any(word in response for word in unity_words)
        }
        
        if any(indicators.values()):
            print(f"  Patterns detected: {[k for k, v in indicators.items() if v]}")
    
    def run_monitoring_session(self, duration_hours=12):
        """Run monitoring session for specified duration"""
        
        print(f"=== Starting Conceptual Monitoring Session ===")
        print(f"Duration: {duration_hours} hours")
        print(f"Probe interval: {self.probe_interval / 60} minutes")
        print(f"Models: {', '.join(self.models)}")
        print(f"\nMonitoring for emergence patterns...\n")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        # Initial probe
        self.run_probe()
        
        # Main monitoring loop
        while time.time() < end_time:
            # Wait for next probe interval
            time.sleep(self.probe_interval)
            
            # Run probe
            self.run_probe()
            
            # Every 10 probes, summarize findings
            if self.probe_count % 10 == 0:
                self.summarize_findings()
        
        print("\n=== Monitoring Session Complete ===")
        self.summarize_findings()
    
    def summarize_findings(self):
        """Summarize monitoring findings"""
        
        print(f"\n--- Summary after {self.probe_count} probes ---")
        
        # Read monitoring log
        try:
            entries = []
            with open("/home/dp/ai-workspace/conceptual_monitoring.jsonl", "r") as f:
                for line in f:
                    entries.append(json.loads(line))
            
            if entries:
                # Calculate statistics
                model_counts = {}
                total_length = 0
                
                for entry in entries[-10:]:  # Last 10 entries
                    model = entry["model"]
                    model_counts[model] = model_counts.get(model, 0) + 1
                    total_length += entry["response_length"]
                
                print(f"Recent activity:")
                for model, count in model_counts.items():
                    print(f"  {model}: {count} probes")
                print(f"  Avg response length: {total_length / len(entries[-10:]):.0f} chars")
        
        except Exception as e:
            print(f"Could not summarize: {str(e)}")


def main():
    """Run conceptual monitoring"""
    
    # Create monitor with 5-minute intervals for testing
    monitor = ConceptualMonitor(probe_interval_minutes=5)
    
    # Run for 1 hour as a test
    monitor.run_monitoring_session(duration_hours=1)
    
    # Log completion
    with open("/home/dp/ai-workspace/autonomous_exploration_log.md", "a") as f:
        f.write(f"\n### Conceptual Monitoring Started - {datetime.now().isoformat()}\n")
        f.write(f"- Running lightweight probes every 5 minutes\n")
        f.write(f"- Monitoring for emergence patterns\n")
        f.write(f"- Results logged to conceptual_monitoring.jsonl\n\n")


if __name__ == "__main__":
    # Note: In production, this would run as a background service
    # For now, running a short test session
    main()