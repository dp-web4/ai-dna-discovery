#!/usr/bin/env python3
"""
Multi-Model Collaboration Experiment
Testing resonance/dissonance patterns between AI models in Web4 context
"""

import json
import time
from typing import Dict, List, Tuple
from ai_lct_ollama_integration import OllamaLCTClient
from ai_lct_experiment import AIEntityManager


class CollaborativeAISession:
    """Manages collaborative sessions between multiple AI models"""
    
    def __init__(self):
        self.client = OllamaLCTClient()
        self.models = []
        self.session_lct = None
        
    def register_models(self, model_names: List[str]):
        """Register multiple models for collaboration"""
        for model in model_names:
            lct = self.client.register_model(model)
            self.models.append((model, lct))
            print(f"Registered {model} for collaboration")
    
    def create_session(self, topic: str) -> str:
        """Create a collaborative session LCT"""
        session_id = f"session_{topic.replace(' ', '_')}_{int(time.time())}"
        self.session_lct = self.client.create_task_lct(f"Collaborative session: {topic}")
        
        # Link all participating models to the session
        for model, lct in self.models:
            self.session_lct.add_link("participants", lct.entity_id)
            lct.add_link("participates_in", self.session_lct.entity_id)
        
        self.client.manager.save_registry()
        return session_id
    
    def detect_resonance(self, response1: str, response2: str) -> Tuple[str, float]:
        """Detect resonance/dissonance between two responses"""
        # Simple heuristic - in reality this would be more sophisticated
        
        # Check for agreement keywords
        agreement_words = ["agree", "similarly", "likewise", "indeed", "correct", "yes"]
        disagreement_words = ["disagree", "however", "but", "contrary", "no", "incorrect"]
        
        r1_lower = response1.lower()
        r2_lower = response2.lower()
        
        # Count agreement/disagreement indicators
        agreement_score = sum(1 for word in agreement_words if word in r2_lower)
        disagreement_score = sum(1 for word in disagreement_words if word in r2_lower)
        
        # Check for conceptual alignment (shared key terms)
        r1_concepts = set(w for w in r1_lower.split() if len(w) > 5)
        r2_concepts = set(w for w in r2_lower.split() if len(w) > 5)
        overlap = len(r1_concepts & r2_concepts) / max(len(r1_concepts), len(r2_concepts), 1)
        
        # Determine resonance type
        if agreement_score > disagreement_score and overlap > 0.3:
            return "resonance", 0.5 + (overlap * 0.5)
        elif disagreement_score > agreement_score:
            return "dissonance", 0.3 - (overlap * 0.2)
        else:
            return "indifference", 0.5
    
    def collaborative_analysis(self, prompt: str, rounds: int = 3) -> Dict:
        """Run a collaborative analysis session"""
        
        if not self.session_lct:
            raise ValueError("No session created. Call create_session first.")
        
        results = {
            "session_id": self.session_lct.entity_id,
            "prompt": prompt,
            "rounds": [],
            "resonance_patterns": []
        }
        
        current_prompt = prompt
        previous_responses = []
        
        for round_num in range(rounds):
            round_data = {
                "round": round_num + 1,
                "responses": [],
                "energy_used": 0
            }
            
            # Get response from each model
            for i, (model, lct) in enumerate(self.models):
                # Build context from previous responses
                if previous_responses:
                    context = f"Previous insights:\n"
                    for prev_model, prev_response in previous_responses[-2:]:
                        context += f"{prev_model}: {prev_response[:200]}...\n"
                    context += f"\nConsidering these perspectives, {current_prompt}"
                else:
                    context = current_prompt
                
                # Get response with energy tracking
                energy_cost = 10 + (round_num * 5)  # Increasing energy cost per round
                result = self.client.generate(model, context, energy_cost)
                
                if "error" not in result:
                    response = result["response"]
                    round_data["responses"].append({
                        "model": model,
                        "response": response,
                        "quality_score": result["quality_score"],
                        "atp_remaining": result["atp_remaining"]
                    })
                    round_data["energy_used"] += energy_cost
                    
                    # Detect resonance with previous responses in this round
                    if i > 0:
                        prev_response = round_data["responses"][i-1]["response"]
                        resonance_type, resonance_score = self.detect_resonance(
                            prev_response, response
                        )
                        results["resonance_patterns"].append({
                            "round": round_num + 1,
                            "models": [round_data["responses"][i-1]["model"], model],
                            "type": resonance_type,
                            "score": resonance_score
                        })
                    
                    previous_responses.append((model, response))
            
            results["rounds"].append(round_data)
            
            # Update prompt for next round based on emerging insights
            if round_num < rounds - 1:
                current_prompt = "Synthesize and expand on the previous insights. What patterns emerge?"
        
        # Calculate overall session coherence
        if results["resonance_patterns"]:
            avg_resonance = sum(p["score"] for p in results["resonance_patterns"]) / len(results["resonance_patterns"])
            results["session_coherence"] = avg_resonance
        else:
            results["session_coherence"] = 0.5
        
        return results
    
    def generate_collaboration_report(self, results: Dict) -> str:
        """Generate a human-readable report of the collaboration"""
        
        report = f"=== Collaborative AI Session Report ===\n"
        report += f"Session: {results['session_id']}\n"
        report += f"Topic: {results['prompt']}\n"
        report += f"Overall Coherence: {results.get('session_coherence', 0):.3f}\n\n"
        
        # Resonance summary
        resonance_count = sum(1 for p in results["resonance_patterns"] if p["type"] == "resonance")
        dissonance_count = sum(1 for p in results["resonance_patterns"] if p["type"] == "dissonance")
        
        report += f"Resonance Patterns:\n"
        report += f"- Resonant exchanges: {resonance_count}\n"
        report += f"- Dissonant exchanges: {dissonance_count}\n"
        report += f"- Total energy expended: {sum(r['energy_used'] for r in results['rounds'])} ATP\n\n"
        
        # Key insights per round
        for round_data in results["rounds"]:
            report += f"Round {round_data['round']}:\n"
            for resp in round_data["responses"]:
                report += f"  {resp['model']} (Q:{resp['quality_score']:.2f}): "
                report += f"{resp['response'][:100]}...\n"
            report += "\n"
        
        return report


def test_collaborative_reasoning():
    """Test collaborative reasoning between models on Synchronism concepts"""
    
    session = CollaborativeAISession()
    
    # Register available models
    session.register_models(["phi3:mini", "tinyllama:latest"])
    
    # Create a session on Synchronism concepts
    session_id = session.create_session("synchronism_intent_analysis")
    
    # Run collaborative analysis
    prompt = """In the Synchronism framework, 'intent' is proposed as the fundamental force 
    of reality, similar to how physics has fundamental forces. How might this concept 
    bridge the gap between consciousness and physical phenomena?"""
    
    print(f"\nStarting collaborative session: {session_id}")
    print(f"Topic: {prompt}\n")
    
    results = session.collaborative_analysis(prompt, rounds=2)
    
    # Generate and save report
    report = session.generate_collaboration_report(results)
    print(report)
    
    # Save detailed results
    with open(f"/home/dp/ai-workspace/collaboration_{session_id}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    test_collaborative_reasoning()