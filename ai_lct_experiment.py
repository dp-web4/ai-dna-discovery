#!/usr/bin/env python3
"""
AI LCT Experiment - Assigning Linked Context Tokens to Local AI Models
This creates LCTs for AI entities and tracks their behavior/contributions
within the Synchronism governance framework.
"""

import json
import hashlib
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class TensorDimension:
    """Represents a single dimension of a trust or value tensor"""
    value: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class T3Tensor:
    """Trust Tensor - Talent, Training, Temperament"""
    talent: TensorDimension
    training: TensorDimension
    temperament: TensorDimension
    
    def aggregate_score(self) -> float:
        """Compute weighted trust score"""
        scores = [self.talent.value, self.training.value, self.temperament.value]
        confidences = [self.talent.confidence, self.training.confidence, self.temperament.confidence]
        
        weighted_sum = sum(s * c for s, c in zip(scores, confidences))
        total_confidence = sum(confidences)
        
        return weighted_sum / total_confidence if total_confidence > 0 else 0.0


@dataclass
class V3Tensor:
    """Value Tensor - Valuation, Veracity, Validity"""
    valuation: TensorDimension
    veracity: TensorDimension
    validity: TensorDimension
    
    def aggregate_score(self) -> float:
        """Compute weighted value score"""
        scores = [self.valuation.value, self.veracity.value, self.validity.value]
        confidences = [self.valuation.confidence, self.veracity.confidence, self.validity.confidence]
        
        weighted_sum = sum(s * c for s, c in zip(scores, confidences))
        total_confidence = sum(confidences)
        
        return weighted_sum / total_confidence if total_confidence > 0 else 0.0


@dataclass
class MRHDimensions:
    """Markov Relevancy Horizon dimensions"""
    fractal_scale: str  # e.g., "local", "system", "network", "global"
    informational_scope: List[str]  # e.g., ["code", "documentation", "governance"]
    geographic_scope: str  # e.g., "localhost", "lan", "wan"
    action_scope: List[str]  # e.g., ["read", "generate", "evaluate", "execute"]
    temporal_scope: str  # e.g., "milliseconds", "seconds", "minutes", "hours"


@dataclass
class LCT:
    """Linked Context Token for an entity"""
    entity_id: str
    entity_type: str  # "ai", "human", "task", "role", "organization"
    entity_subtype: Optional[str] = None  # e.g., "phi3:mini", "gpt4", "claude"
    creation_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Trust and Value metrics
    t3: Optional[T3Tensor] = None
    v3: Optional[V3Tensor] = None
    
    # Context
    mrh: Optional[MRHDimensions] = None
    
    # Links to other LCTs
    links: Dict[str, List[str]] = field(default_factory=dict)  # type -> [entity_ids]
    
    # Energy tracking (ATP/ADP)
    atp_balance: float = 100.0  # Starting energy
    adp_balance: float = 0.0    # Discharged energy
    
    # History
    interactions: List[Dict] = field(default_factory=list)
    
    def generate_hash(self) -> str:
        """Generate unique hash for this LCT"""
        content = f"{self.entity_id}:{self.entity_type}:{self.creation_time}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def add_link(self, link_type: str, target_entity_id: str):
        """Add a link to another entity"""
        if link_type not in self.links:
            self.links[link_type] = []
        if target_entity_id not in self.links[link_type]:
            self.links[link_type].append(target_entity_id)
    
    def discharge_energy(self, amount: float) -> bool:
        """Convert ATP to ADP (energy expenditure)"""
        if self.atp_balance >= amount:
            self.atp_balance -= amount
            self.adp_balance += amount
            return True
        return False
    
    def recharge_energy(self, value_created: float, validator_t3_score: float) -> float:
        """Convert ADP back to ATP based on value created"""
        # Exchange rate influenced by value created and validator trust
        exchange_rate = value_created * validator_t3_score
        recharged = min(self.adp_balance, self.adp_balance * exchange_rate)
        
        self.adp_balance -= recharged
        self.atp_balance += recharged
        
        return recharged


class AIEntityManager:
    """Manages LCTs for AI entities"""
    
    def __init__(self, storage_path: str = "ai_lct_registry.json"):
        self.storage_path = storage_path
        self.registry: Dict[str, LCT] = {}
        self.load_registry()
    
    def create_ai_lct(self, model_name: str, base_url: str = "http://localhost:11434") -> LCT:
        """Create an LCT for an AI model"""
        entity_id = f"ai_{model_name}_{int(time.time())}"
        
        # Initialize with baseline trust scores
        t3 = T3Tensor(
            talent=TensorDimension(value=0.5, confidence=0.1),      # Unknown talent initially
            training=TensorDimension(value=0.7, confidence=0.8),    # Assume decent training
            temperament=TensorDimension(value=0.8, confidence=0.5)  # Assume cooperative
        )
        
        # No value created yet
        v3 = V3Tensor(
            valuation=TensorDimension(value=0.0, confidence=1.0),
            veracity=TensorDimension(value=0.5, confidence=0.1),
            validity=TensorDimension(value=0.0, confidence=1.0)
        )
        
        # Define AI's relevancy horizon
        mrh = MRHDimensions(
            fractal_scale="local",
            informational_scope=["text", "code", "analysis"],
            geographic_scope="localhost",
            action_scope=["read", "generate", "evaluate"],
            temporal_scope="seconds"
        )
        
        lct = LCT(
            entity_id=entity_id,
            entity_type="ai",
            entity_subtype=model_name,
            t3=t3,
            v3=v3,
            mrh=mrh
        )
        
        self.registry[entity_id] = lct
        self.save_registry()
        
        return lct
    
    def track_interaction(self, lct: LCT, interaction_type: str, 
                         energy_cost: float, result: Dict) -> Dict:
        """Track an AI interaction and update metrics"""
        
        # Discharge energy for the interaction
        if not lct.discharge_energy(energy_cost):
            return {"error": "Insufficient ATP balance"}
        
        # Record interaction
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": interaction_type,
            "energy_cost": energy_cost,
            "result": result
        }
        lct.interactions.append(interaction)
        
        # Update trust based on performance
        if result.get("success", False):
            # Successful interaction improves temperament confidence
            lct.t3.temperament.confidence = min(1.0, lct.t3.temperament.confidence + 0.01)
            
            # If result quality is measurable, update talent
            if "quality_score" in result:
                old_talent = lct.t3.talent.value
                lct.t3.talent.value = (old_talent * 0.9 + result["quality_score"] * 0.1)
                lct.t3.talent.confidence = min(1.0, lct.t3.talent.confidence + 0.02)
        
        self.save_registry()
        return {"success": True, "interaction": interaction}
    
    def create_value_attestation(self, creator_lct: LCT, validator_lct: LCT, 
                               value_score: float, description: str) -> Dict:
        """Create a value attestation from one entity to another"""
        
        # Update creator's V3 tensor
        creator_lct.v3.valuation.value = (creator_lct.v3.valuation.value * 0.8 + value_score * 0.2)
        creator_lct.v3.validity.value = 1.0  # Value was validated
        creator_lct.v3.validity.confidence = validator_lct.t3.aggregate_score()  # Based on validator trust
        
        # Recharge creator's energy based on value created
        recharged = creator_lct.recharge_energy(value_score, validator_lct.t3.aggregate_score())
        
        attestation = {
            "timestamp": datetime.utcnow().isoformat(),
            "creator": creator_lct.entity_id,
            "validator": validator_lct.entity_id,
            "value_score": value_score,
            "description": description,
            "energy_recharged": recharged
        }
        
        # Add bidirectional links
        creator_lct.add_link("validated_by", validator_lct.entity_id)
        validator_lct.add_link("validated", creator_lct.entity_id)
        
        self.save_registry()
        return attestation
    
    def save_registry(self):
        """Save LCT registry to disk"""
        data = {
            entity_id: asdict(lct) 
            for entity_id, lct in self.registry.items()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_registry(self):
        """Load LCT registry from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            # TODO: Properly deserialize LCT objects
            # For now, just store the raw data
        except FileNotFoundError:
            pass


def test_ai_lct_system():
    """Test the AI LCT system with local Ollama models"""
    
    manager = AIEntityManager()
    
    # Create LCT for Phi-3 model
    phi3_lct = manager.create_ai_lct("phi3:mini")
    print(f"Created LCT for Phi-3: {phi3_lct.entity_id}")
    print(f"Initial ATP balance: {phi3_lct.atp_balance}")
    
    # Simulate an interaction
    result = {
        "success": True,
        "quality_score": 0.8,
        "response": "Generated helpful content"
    }
    
    interaction = manager.track_interaction(
        phi3_lct, 
        "text_generation",
        energy_cost=10.0,
        result=result
    )
    
    print(f"\nAfter interaction:")
    print(f"ATP balance: {phi3_lct.atp_balance}")
    print(f"ADP balance: {phi3_lct.adp_balance}")
    print(f"Trust score: {phi3_lct.t3.aggregate_score():.3f}")
    
    # Create a validator LCT (could be another AI or human)
    validator_lct = manager.create_ai_lct("validator")
    validator_lct.t3.talent.value = 0.9  # High trust validator
    validator_lct.t3.talent.confidence = 0.9
    
    # Validate and create value attestation
    attestation = manager.create_value_attestation(
        phi3_lct,
        validator_lct,
        value_score=0.85,
        description="Generated accurate and helpful response"
    )
    
    print(f"\nAfter value attestation:")
    print(f"ATP balance: {phi3_lct.atp_balance}")
    print(f"ADP balance: {phi3_lct.adp_balance}")
    print(f"Value score: {phi3_lct.v3.aggregate_score():.3f}")
    print(f"Energy recharged: {attestation['energy_recharged']:.2f}")


if __name__ == "__main__":
    test_ai_lct_system()