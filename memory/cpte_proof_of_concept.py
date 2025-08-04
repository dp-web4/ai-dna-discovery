#!/usr/bin/env python3
"""
Proof of concept for Contextual Pretrained Experts (CPTEs)
Demonstrates knowledge aging, markers, and external consultation
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Any
import json
import time

@dataclass
class CPTEMarker:
    """Lightweight marker for aged-out knowledge"""
    domain: str
    last_used: datetime
    confidence: float
    external_ref: Optional[str] = None
    
    def to_dict(self):
        return {
            'domain': self.domain,
            'last_used': self.last_used.isoformat(),
            'confidence': self.confidence,
            'external_ref': self.external_ref
        }

class InternalCPTE:
    """Internal expertise that's actively maintained"""
    def __init__(self, domain: str, knowledge: Dict[str, Any]):
        self.domain = domain
        self.knowledge = knowledge
        self.confidence = 0.9
        self.usage_count = 0
        self.last_accessed = datetime.now()
        
    def process(self, query: str) -> str:
        """Process query using internal knowledge"""
        self.usage_count += 1
        self.last_accessed = datetime.now()
        
        # Simple keyword matching for demo
        for key, value in self.knowledge.items():
            if key.lower() in query.lower():
                return f"[Internal {self.domain}]: {value}"
                
        return f"[Internal {self.domain}]: No specific knowledge about '{query}'"
        
    def decay_confidence(self) -> float:
        """Calculate decayed confidence based on time since last use"""
        age = datetime.now() - self.last_accessed
        days_old = age.days
        
        # Exponential decay: loses 10% confidence per month of non-use
        decay_rate = 0.9 ** (days_old / 30)
        return self.confidence * decay_rate

class ExternalCPTE:
    """Simulated external expert service"""
    def __init__(self, domain: str):
        self.domain = domain
        self.knowledge_base = {
            'differential_equations': {
                'solve': 'To solve dy/dx = f(x), integrate both sides',
                'linear': 'Linear DE: y\' + P(x)y = Q(x)',
                'separable': 'Separable: dy/dx = g(x)h(y) → ∫dy/h(y) = ∫g(x)dx'
            },
            'legal_contracts': {
                'nda': 'NDA requires: parties, confidential info definition, term',
                'liability': 'Limitation of liability clauses cap damages',
                'termination': 'Include termination conditions and notice period'
            }
        }
        
    def consult(self, query: str) -> str:
        """Simulate external consultation"""
        time.sleep(0.5)  # Simulate network delay
        
        if self.domain in self.knowledge_base:
            kb = self.knowledge_base[self.domain]
            for key, value in kb.items():
                if key in query.lower():
                    return f"[External {self.domain} Expert]: {value}"
                    
        return f"[External {self.domain} Expert]: General advice for '{query}'"

class CPTEManager:
    """Manages internal/external expertise lifecycle"""
    
    def __init__(self):
        self.internal_cptes: Dict[str, InternalCPTE] = {}
        self.knowledge_markers: Dict[str, CPTEMarker] = {}
        self.external_services: Dict[str, ExternalCPTE] = {}
        
        # Usage threshold - if used less than this, decay to marker
        self.usage_threshold = 3
        self.confidence_threshold = 0.5
        
    def add_internal_knowledge(self, domain: str, knowledge: Dict[str, Any]):
        """Add or update internal expertise"""
        self.internal_cptes[domain] = InternalCPTE(domain, knowledge)
        print(f"✓ Added internal CPTE for '{domain}'")
        
    def register_external_service(self, domain: str, service_ref: str):
        """Register an external expert service"""
        self.external_services[service_ref] = ExternalCPTE(domain)
        print(f"✓ Registered external service '{service_ref}' for '{domain}'")
        
    def query_knowledge(self, domain: str, query: str) -> Optional[str]:
        """Query knowledge - internal, external, or none"""
        print(f"\n🔍 Querying '{domain}': {query}")
        
        # Check internal CPTE first
        if domain in self.internal_cptes:
            cpte = self.internal_cptes[domain]
            print(f"  → Using internal CPTE (usage: {cpte.usage_count}, conf: {cpte.confidence:.2f})")
            return cpte.process(query)
            
        # Check knowledge markers
        if domain in self.knowledge_markers:
            marker = self.knowledge_markers[domain]
            print(f"  → Found marker (conf: {marker.confidence:.2f}, last used: {marker.last_used.date()})")
            
            if marker.external_ref and marker.external_ref in self.external_services:
                print(f"  → Consulting external expert '{marker.external_ref}'...")
                return self.external_services[marker.external_ref].consult(query)
            else:
                return f"Knowledge marker exists but no external service available"
                
        return f"No knowledge available for domain '{domain}'"
        
    def age_out_unused_knowledge(self):
        """Convert unused internal CPTEs to markers"""
        print("\n🕐 Aging out unused knowledge...")
        
        to_remove = []
        for domain, cpte in self.internal_cptes.items():
            current_confidence = cpte.decay_confidence()
            
            if cpte.usage_count < self.usage_threshold or current_confidence < self.confidence_threshold:
                # Create marker
                marker = CPTEMarker(
                    domain=domain,
                    last_used=cpte.last_accessed,
                    confidence=current_confidence,
                    external_ref=f"{domain}_expert_service"
                )
                
                self.knowledge_markers[domain] = marker
                to_remove.append(domain)
                
                print(f"  → Aged out '{domain}' to marker (usage: {cpte.usage_count}, conf: {current_confidence:.2f})")
                
        # Remove aged-out internal CPTEs
        for domain in to_remove:
            del self.internal_cptes[domain]
            
    def show_status(self):
        """Display current knowledge status"""
        print("\n📊 Knowledge Status:")
        print(f"  Internal CPTEs: {len(self.internal_cptes)}")
        for domain, cpte in self.internal_cptes.items():
            print(f"    - {domain}: {cpte.usage_count} uses, conf: {cpte.confidence:.2f}")
            
        print(f"  Knowledge Markers: {len(self.knowledge_markers)}")
        for domain, marker in self.knowledge_markers.items():
            print(f"    - {domain}: last used {marker.last_used.date()}, conf: {marker.confidence:.2f}")
            
        print(f"  External Services: {len(self.external_services)}")

def demo_cpte_lifecycle():
    """Demonstrate CPTE lifecycle"""
    print("=== CPTE Lifecycle Demo ===\n")
    
    manager = CPTEManager()
    
    # Phase 1: Active internal knowledge
    print("📚 Phase 1: Loading internal knowledge...")
    
    # Frequently used knowledge (programming)
    manager.add_internal_knowledge('python', {
        'async': 'Use async/await for concurrent operations',
        'decorator': 'Decorators wrap functions to modify behavior',
        'list comprehension': '[x*2 for x in range(10)]'
    })
    
    # Occasionally used knowledge (math)
    manager.add_internal_knowledge('calculus', {
        'derivative': 'Rate of change: d/dx[f(x)]',
        'integral': 'Area under curve: ∫f(x)dx',
        'chain rule': 'd/dx[f(g(x))] = f\'(g(x)) * g\'(x)'
    })
    
    # Register external services
    manager.register_external_service('differential_equations', 'diff_eq_expert_service')
    manager.register_external_service('legal_contracts', 'legal_expert_service')
    
    manager.show_status()
    
    # Phase 2: Use internal knowledge
    print("\n📖 Phase 2: Using internal knowledge...")
    
    # Use Python knowledge frequently
    for _ in range(4):
        result = manager.query_knowledge('python', 'How do I use async in Python?')
        print(f"  {result}")
        
    # Use calculus rarely
    result = manager.query_knowledge('calculus', 'What is a derivative?')
    print(f"  {result}")
    
    # Phase 3: Age out unused knowledge
    print("\n⏳ Phase 3: Simulating time passage...")
    
    # Simulate aging by modifying last_accessed
    if 'calculus' in manager.internal_cptes:
        # Make calculus knowledge "old"
        manager.internal_cptes['calculus'].last_accessed = datetime.now() - timedelta(days=180)
        manager.internal_cptes['calculus'].usage_count = 1
        
    manager.age_out_unused_knowledge()
    manager.show_status()
    
    # Phase 4: Query aged-out knowledge
    print("\n🔮 Phase 4: Querying aged-out knowledge...")
    
    # Python still internal
    result = manager.query_knowledge('python', 'What is a decorator?')
    print(f"  {result}")
    
    # Calculus now external
    result = manager.query_knowledge('calculus', 'What is the chain rule?')
    print(f"  {result}")
    
    # Query knowledge we never had internally
    result = manager.query_knowledge('differential_equations', 'How to solve separable equations?')
    print(f"  {result}")
    
    # Phase 5: Demonstrate missing knowledge
    print("\n❓ Phase 5: Querying unknown domain...")
    result = manager.query_knowledge('quantum_physics', 'What is superposition?')
    print(f"  {result}")

if __name__ == "__main__":
    demo_cpte_lifecycle()