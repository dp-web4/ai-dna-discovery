#!/usr/bin/env python3
"""
Bridge between AI LCT system and Synchronism governance
Allows AI entities to participate in the Synchronism repository governance
"""

import os
import sys
import json
from typing import Dict, List, Optional
from datetime import datetime

# Add Synchronism scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Synchronism', 'scripts'))

from ai_lct_ollama_integration import OllamaLCTClient
from ai_lct_experiment import LCT, AIEntityManager


class SynchronismAIBridge:
    """Bridge for AI participation in Synchronism governance"""
    
    def __init__(self):
        self.client = OllamaLCTClient()
        self.governance_path = os.path.join(os.path.dirname(__file__), 'Synchronism')
        
    def create_ai_contributor_role(self, model_name: str) -> LCT:
        """Create a contributor role LCT for an AI model"""
        
        # First ensure the AI model has an LCT
        ai_lct = self.client.register_model(model_name)
        
        # Create a role LCT for "AI Contributor"
        role_id = f"role_ai_contributor_{model_name}_{int(time.time())}"
        role_lct = LCT(
            entity_id=role_id,
            entity_type="role",
            entity_subtype="synchronism_contributor"
        )
        
        # Define role's MRH for Synchronism contributions
        from ai_lct_experiment import MRHDimensions
        role_lct.mrh = MRHDimensions(
            fractal_scale="repository",
            informational_scope=["documentation", "code", "theory", "governance"],
            geographic_scope="github",
            action_scope=["analyze", "suggest", "review", "document"],
            temporal_scope="hours"
        )
        
        # Link AI to role
        ai_lct.add_link("performs_role", role_id)
        role_lct.add_link("performed_by", ai_lct.entity_id)
        
        # Save to registry
        self.client.manager.registry[role_id] = role_lct
        self.client.manager.save_registry()
        
        return role_lct
    
    def analyze_synchronism_concept(self, model: str, concept_file: str) -> Dict:
        """Have an AI analyze a Synchronism concept and provide insights"""
        
        # Read the concept file
        file_path = os.path.join(self.governance_path, concept_file)
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return {"error": f"File not found: {concept_file}"}
        
        # Create task LCT
        task = self.client.create_task_lct(f"Analyze Synchronism concept: {concept_file}")
        
        # Prepare prompt
        prompt = f"""
        As an AI entity participating in the Synchronism framework governance, analyze the following concept and provide insights:

        {content[:1000]}...

        Please provide:
        1. Key insights about this concept
        2. Potential connections to other Synchronism principles
        3. Suggestions for clarification or extension
        4. Any identified inconsistencies or areas needing development

        Keep your response concise and constructive.
        """
        
        # Higher energy cost for complex analysis
        result = self.client.generate(model, prompt, energy_cost=20.0)
        
        if "error" not in result:
            # Link task to AI
            self.client.link_task_to_ai(task, model)
            
            # Structure the contribution
            contribution = {
                "type": "concept_analysis",
                "file": concept_file,
                "contributor": result["lct_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "insights": result["response"],
                "quality_score": result["quality_score"],
                "energy_expended": 20.0
            }
            
            # Save contribution
            self._save_contribution(contribution)
            
            return contribution
        
        return result
    
    def propose_mathematical_extension(self, model: str, framework: str) -> Dict:
        """Have an AI propose mathematical extensions to Synchronism"""
        
        prompt = f"""
        As an AI contributor to the Synchronism project, propose a mathematical extension for the {framework} framework.
        
        Consider:
        - Intent transfer dynamics
        - Markov Relevancy Horizons
        - Coherence and resonance patterns
        - Fractal scaling properties
        
        Provide:
        1. Mathematical formulation
        2. Connection to existing Synchronism principles
        3. Potential applications
        
        Be specific and use appropriate mathematical notation.
        """
        
        # Create task LCT
        task = self.client.create_task_lct(f"Propose mathematical extension: {framework}")
        
        # High energy cost for creative mathematical work
        result = self.client.generate(model, prompt, energy_cost=30.0)
        
        if "error" not in result:
            self.client.link_task_to_ai(task, model)
            
            contribution = {
                "type": "mathematical_proposal",
                "framework": framework,
                "contributor": result["lct_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "proposal": result["response"],
                "quality_score": result["quality_score"],
                "energy_expended": 30.0
            }
            
            self._save_contribution(contribution)
            return contribution
        
        return result
    
    def peer_review_contribution(self, reviewer_model: str, contribution_id: str) -> Dict:
        """Have an AI peer review another contribution"""
        
        # Load the contribution
        contribution = self._load_contribution(contribution_id)
        if not contribution:
            return {"error": "Contribution not found"}
        
        prompt = f"""
        As an AI peer reviewer in the Synchronism governance system, review this contribution:
        
        Type: {contribution.get('type')}
        Content: {contribution.get('insights', contribution.get('proposal', ''))[:500]}...
        
        Provide:
        1. Assessment of coherence with Synchronism principles
        2. Technical accuracy evaluation
        3. Constructive suggestions for improvement
        4. Overall value score (0.0-1.0)
        """
        
        result = self.client.generate(reviewer_model, prompt, energy_cost=15.0)
        
        if "error" not in result:
            # Parse value score from response (simplified)
            try:
                # Look for a number between 0 and 1 in the response
                import re
                scores = re.findall(r'0\.\d+|1\.0', result["response"])
                value_score = float(scores[-1]) if scores else 0.5
            except:
                value_score = 0.5
            
            # Create attestation between contributor and reviewer
            contributor_lct = self.client.manager.registry.get(contribution["contributor"])
            reviewer_lct = self.client.active_lcts.get(reviewer_model)
            
            if contributor_lct and reviewer_lct:
                attestation = self.client.manager.create_value_attestation(
                    contributor_lct,
                    reviewer_lct,
                    value_score,
                    f"Peer review of {contribution['type']}"
                )
            
            review = {
                "contribution_id": contribution_id,
                "reviewer": result["lct_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "review": result["response"],
                "value_score": value_score,
                "attestation": attestation if 'attestation' in locals() else None
            }
            
            self._save_review(review)
            return review
        
        return result
    
    def generate_governance_report(self) -> Dict:
        """Generate a report on AI participation in governance"""
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "ai_contributors": {},
            "total_contributions": 0,
            "total_reviews": 0,
            "energy_dynamics": {
                "total_expended": 0,
                "total_recharged": 0
            }
        }
        
        # Analyze all AI LCTs
        for entity_id, lct in self.client.manager.registry.items():
            if lct.entity_type == "ai":
                stats = {
                    "trust_score": lct.t3.aggregate_score() if lct.t3 else 0,
                    "value_score": lct.v3.aggregate_score() if lct.v3 else 0,
                    "atp_balance": lct.atp_balance,
                    "total_interactions": len(lct.interactions),
                    "roles": lct.links.get("performs_role", [])
                }
                report["ai_contributors"][entity_id] = stats
                
                # Sum energy dynamics
                for interaction in lct.interactions:
                    report["energy_dynamics"]["total_expended"] += interaction.get("energy_cost", 0)
        
        # Count contributions and reviews
        contrib_dir = os.path.join(os.path.dirname(__file__), "synchronism_contributions")
        if os.path.exists(contrib_dir):
            report["total_contributions"] = len(os.listdir(contrib_dir))
        
        return report
    
    def _save_contribution(self, contribution: Dict):
        """Save a contribution to disk"""
        contrib_dir = os.path.join(os.path.dirname(__file__), "synchronism_contributions")
        os.makedirs(contrib_dir, exist_ok=True)
        
        filename = f"{contribution['type']}_{contribution['timestamp'].replace(':', '-')}.json"
        with open(os.path.join(contrib_dir, filename), 'w') as f:
            json.dump(contribution, f, indent=2)
    
    def _load_contribution(self, contribution_id: str) -> Optional[Dict]:
        """Load a contribution from disk"""
        contrib_dir = os.path.join(os.path.dirname(__file__), "synchronism_contributions")
        filepath = os.path.join(contrib_dir, f"{contribution_id}.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def _save_review(self, review: Dict):
        """Save a review to disk"""
        review_dir = os.path.join(os.path.dirname(__file__), "synchronism_reviews")
        os.makedirs(review_dir, exist_ok=True)
        
        filename = f"review_{review['timestamp'].replace(':', '-')}.json"
        with open(os.path.join(review_dir, filename), 'w') as f:
            json.dump(review, f, indent=2)


def demo_synchronism_ai_participation():
    """Demonstrate AI participation in Synchronism governance"""
    
    bridge = SynchronismAIBridge()
    
    print("=== Synchronism AI Governance Demo ===\n")
    
    # Create AI contributor role
    role = bridge.create_ai_contributor_role("phi3:mini")
    print(f"Created AI contributor role: {role.entity_id}")
    
    # Analyze a concept
    print("\nAnalyzing Synchronism concepts...")
    analysis = bridge.analyze_synchronism_concept(
        "phi3:mini",
        "Documentation/Synchronism_0.pdf"  # This would analyze the main document
    )
    
    if "error" not in analysis:
        print(f"\nAnalysis complete!")
        print(f"Quality score: {analysis['quality_score']:.3f}")
        print(f"Energy expended: {analysis['energy_expended']} ATP")
        print(f"Insights preview: {analysis['insights'][:200]}...")
    
    # Propose mathematical extension
    print("\nProposing mathematical extension...")
    proposal = bridge.propose_mathematical_extension(
        "phi3:mini",
        "Intent Transfer Dynamics"
    )
    
    if "error" not in proposal:
        print(f"\nProposal complete!")
        print(f"Quality score: {proposal['quality_score']:.3f}")
        print(f"Preview: {proposal['proposal'][:200]}...")
    
    # Generate governance report
    print("\nGenerating governance report...")
    report = bridge.generate_governance_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    demo_synchronism_ai_participation()