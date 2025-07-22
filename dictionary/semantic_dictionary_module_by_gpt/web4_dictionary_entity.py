#!/usr/bin/env python3
"""
Web4 Dictionary Entity Manager
Implements trust-based consensus and LCT principles for semantic dictionaries
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os

@dataclass
class TrustVector:
    """Trust vector for dictionary entries"""
    source: str  # human-curated, ai-generated, consensus-derived
    weight: float  # 0.0 to 1.0
    validators: List[str] = None  # List of validator IDs
    timestamp: str = None
    
    def __post_init__(self):
        if self.validators is None:
            self.validators = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + 'Z'

@dataclass
class ConsensusRecord:
    """Consensus tracking for dictionary updates"""
    proposal_id: str
    proposed_by: str
    action: str  # add, modify, remove
    target_glyph: str
    new_data: Dict
    votes: Dict[str, bool] = None  # validator_id: vote
    status: str = "pending"  # pending, approved, rejected
    timestamp: str = None
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + 'Z'

class Web4DictionaryEntity:
    """Web4-native dictionary with LCT principles"""
    
    def __init__(self, entity_id: str = "web4.dictionary.unified"):
        self.entity_id = entity_id
        self.entity_file = os.path.join(os.path.dirname(__file__), "dictionary_entity.json")
        self.consensus_log = os.path.join(os.path.dirname(__file__), "consensus_log.json")
        self.load_entity()
        self.load_consensus_log()
        
    def load_entity(self):
        """Load dictionary entity metadata"""
        if os.path.exists(self.entity_file):
            with open(self.entity_file, 'r') as f:
                self.entity = json.load(f)
        else:
            self.initialize_entity()
    
    def load_consensus_log(self):
        """Load consensus history"""
        if os.path.exists(self.consensus_log):
            with open(self.consensus_log, 'r') as f:
                log_data = json.load(f)
                self.consensus_records = [
                    ConsensusRecord(**record) for record in log_data['records']
                ]
        else:
            self.consensus_records = []
    
    def initialize_entity(self):
        """Initialize new Web4 dictionary entity"""
        self.entity = {
            "id": self.entity_id,
            "version": "1.0.0",
            "lct": {
                "issuer": "unified.dictionary.bridge",
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "permissions": ["read", "extend", "link", "validate"],
                "consensus": {
                    "T3": {
                        "threshold": 0.66,  # 2/3 majority
                        "validators": ["human", "ai_claude", "ai_gpt", "ai_gemma"]
                    },
                    "V3": {
                        "verification_layers": 3,
                        "trust_threshold": 0.8
                    }
                },
                "locality": {
                    "primary_node": "tomato",
                    "edge_nodes": ["sprout"],
                    "sync_interval": 3600  # seconds
                }
            },
            "linked_chains": [
                "metaLINXX",
                "Web4-seed",
                "phoenician-lora",
                "consciousness-notation"
            ],
            "description": "Unified semantic dictionary with trust-based consensus",
            "status": "active"
        }
        self.save_entity()
    
    def save_entity(self):
        """Save entity metadata"""
        with open(self.entity_file, 'w') as f:
            json.dump(self.entity, f, indent=2)
    
    def save_consensus_log(self):
        """Save consensus history"""
        log_data = {
            "entity_id": self.entity_id,
            "last_updated": datetime.utcnow().isoformat() + 'Z',
            "records": [asdict(record) for record in self.consensus_records]
        }
        with open(self.consensus_log, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def calculate_trust(self, trust_vectors: List[TrustVector]) -> float:
        """Calculate aggregate trust score using V3 verification"""
        if not trust_vectors:
            return 0.0
        
        # Layer 1: Source credibility
        source_weights = {
            'human-curated': 1.0,
            'human-verified': 0.95,
            'consensus-derived': 0.9,
            'ai-designed': 0.85,
            'ai-generated': 0.8,
            'inferred': 0.7
        }
        
        # Layer 2: Validator consensus
        total_weight = 0.0
        weighted_sum = 0.0
        
        for tv in trust_vectors:
            source_mult = source_weights.get(tv.source, 0.5)
            validator_mult = 1.0 + (len(tv.validators) * 0.1)  # Bonus for multiple validators
            
            final_weight = tv.weight * source_mult * min(validator_mult, 1.5)
            weighted_sum += final_weight
            total_weight += 1.0
        
        # Layer 3: Temporal decay (newer = more trusted)
        base_trust = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Apply V3 threshold
        v3_threshold = self.entity['lct']['consensus'].get('V3', {}).get('trust_threshold', 0.8)
        return base_trust if base_trust >= v3_threshold else base_trust * 0.8
    
    def propose_change(self, action: str, glyph: str, data: Dict, proposer: str) -> str:
        """Propose a dictionary change requiring consensus"""
        proposal_id = hashlib.sha256(
            f"{action}:{glyph}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        record = ConsensusRecord(
            proposal_id=proposal_id,
            proposed_by=proposer,
            action=action,
            target_glyph=glyph,
            new_data=data
        )
        
        self.consensus_records.append(record)
        self.save_consensus_log()
        
        # Auto-approve if proposer is trusted
        if proposer in ['human', 'human-curated']:
            self.vote_on_proposal(proposal_id, proposer, True)
        
        return proposal_id
    
    def vote_on_proposal(self, proposal_id: str, validator: str, vote: bool):
        """Cast vote on a proposal"""
        for record in self.consensus_records:
            if record.proposal_id == proposal_id:
                record.votes[validator] = vote
                
                # Check if consensus reached
                validators = self.entity['lct']['consensus'].get('T3', {}).get('validators', ['human', 'ai_claude', 'ai_gpt', 'ai_gemma'])
                threshold = self.entity['lct']['consensus'].get('T3', {}).get('threshold', 0.66)
                
                yes_votes = sum(1 for v in record.votes.values() if v)
                total_possible = len(validators)
                
                if yes_votes / total_possible >= threshold:
                    record.status = "approved"
                    self.apply_proposal(record)
                elif (total_possible - yes_votes) / total_possible > (1 - threshold):
                    record.status = "rejected"
                
                self.save_consensus_log()
                break
    
    def apply_proposal(self, record: ConsensusRecord):
        """Apply approved proposal to dictionary"""
        # This would update the actual dictionary files
        print(f"Applying approved proposal {record.proposal_id}:")
        print(f"  Action: {record.action}")
        print(f"  Glyph: {record.target_glyph}")
        print(f"  Data: {record.new_data}")
    
    def get_consensus_status(self) -> Dict:
        """Get current consensus statistics"""
        total = len(self.consensus_records)
        approved = sum(1 for r in self.consensus_records if r.status == "approved")
        rejected = sum(1 for r in self.consensus_records if r.status == "rejected")
        pending = sum(1 for r in self.consensus_records if r.status == "pending")
        
        return {
            "total_proposals": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "approval_rate": approved / total if total > 0 else 0.0,
            "active_validators": self.entity['lct']['consensus'].get('T3', {}).get('validators', [])
        }
    
    def sync_with_edge(self, node_id: str) -> Dict:
        """Simulate sync with edge node"""
        return {
            "node": node_id,
            "status": "simulated",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "message": f"Would sync dictionary state with {node_id}"
        }

def demonstrate_web4_entity():
    """Demonstrate Web4 dictionary entity features"""
    print("=== Web4 Dictionary Entity Demo ===\n")
    
    # Initialize entity
    entity = Web4DictionaryEntity()
    
    # 1. Show entity structure
    print("1. Dictionary Entity Structure:")
    print("-" * 40)
    print(f"ID: {entity.entity['id']}")
    print(f"Version: {entity.entity['version']}")
    print(f"Status: {entity.entity['status']}")
    print(f"Validators: {entity.entity['lct']['consensus'].get('T3', {}).get('validators', 'Not configured')}")
    
    # 2. Test trust calculation
    print("\n2. Trust Vector Calculation:")
    print("-" * 40)
    trust_vectors = [
        TrustVector("human-curated", 0.95, ["human"]),
        TrustVector("ai-generated", 0.85, ["ai_claude", "ai_gpt"]),
        TrustVector("consensus-derived", 0.9, ["human", "ai_claude", "ai_gemma"])
    ]
    
    for tv in trust_vectors:
        print(f"  {tv.source}: {tv.weight} (validators: {len(tv.validators)})")
    
    aggregate_trust = entity.calculate_trust(trust_vectors)
    print(f"\nAggregate trust: {aggregate_trust:.3f}")
    
    # 3. Test consensus mechanism
    print("\n3. Consensus Mechanism:")
    print("-" * 40)
    
    # Propose adding a new symbol
    proposal_id = entity.propose_change(
        action="add",
        glyph="🌱",
        data={
            "name": "growth",
            "concept": "emergence/development",
            "system": "extended_notation"
        },
        proposer="human"
    )
    print(f"Created proposal: {proposal_id}")
    
    # Simulate votes
    entity.vote_on_proposal(proposal_id, "ai_claude", True)
    entity.vote_on_proposal(proposal_id, "ai_gpt", True)
    
    # Show consensus status
    status = entity.get_consensus_status()
    print(f"\nConsensus Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 4. Test edge sync
    print("\n4. Edge Node Sync:")
    print("-" * 40)
    sync_result = entity.sync_with_edge("sprout")
    print(f"Sync result: {sync_result}")
    
    print("\n✅ Web4 Dictionary Entity initialized and tested!")

if __name__ == "__main__":
    demonstrate_web4_entity()