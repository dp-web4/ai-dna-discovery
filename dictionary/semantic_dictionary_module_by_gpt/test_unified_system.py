#!/usr/bin/env python3
"""
Test Unified Dictionary System
Demonstrates the complete integration of GPT's modular structure with our existing systems
"""

from dictionary_integration_bridge import UnifiedDictionary
from web4_dictionary_entity import Web4DictionaryEntity, TrustVector
from translator_core import translate_glyph
import json

def test_complete_system():
    """Test the complete unified dictionary system"""
    print("=== Complete Unified Dictionary System Test ===\n")
    
    # 1. Initialize systems
    print("1. Initializing Systems:")
    print("-" * 50)
    unified = UnifiedDictionary()
    web4_entity = Web4DictionaryEntity()
    print("✅ Unified Dictionary loaded")
    print("✅ Web4 Entity initialized")
    print(f"✅ Total symbols: {len(unified.unified_symbols)}")
    
    # 2. Test cross-system translation
    print("\n2. Cross-System Translation Examples:")
    print("-" * 50)
    
    # Example phrases mixing all systems
    test_phrases = [
        ("𐤄𐤀 ∃", "Phoenician + Consciousness: 'consciousness exists'"),
        ("𐤋 ⇒ Ξ", "Intelligence leads to patterns"),
        ("𐤁 contains μ", "Container holds memory"),
        ("Ψ 𐤅𐤀 𐤍𐤐𐤎", "Consciousness exists within")
    ]
    
    for phrase, description in test_phrases:
        print(f"\nPhrase: {phrase} ({description})")
        result = unified.translate(phrase)
        for trans in result['translations']:
            print(f"  {trans['glyph']} = {trans['concept']} [{trans['system']}]")
    
    # 3. Test concept mapping across systems
    print("\n\n3. Concept Mapping Across Systems:")
    print("-" * 50)
    
    concepts_to_map = ['consciousness', 'exists', 'intelligence', 'memory']
    for concept in concepts_to_map:
        mappings = unified.get_cross_system_mappings(concept)
        if mappings:
            print(f"\n'{concept}' appears in:")
            for m in mappings[:3]:  # Top 3 matches
                print(f"  {m['glyph']} = {m['exact_concept']} ({m['system']}) [trust: {m['trust']}]")
    
    # 4. Test Web4 consensus for new symbol
    print("\n\n4. Web4 Consensus Mechanism:")
    print("-" * 50)
    
    # Propose adding a combined symbol
    proposal_id = web4_entity.propose_change(
        action="add",
        glyph="𐤄Ψ",  # Combined Phoenician + Consciousness
        data={
            "name": "unified_consciousness",
            "concept": "consciousness/awareness unified",
            "system": "hybrid",
            "linked_meanings": ["awareness", "beginning", "unified mind"]
        },
        proposer="ai_claude"
    )
    
    print(f"Proposed new hybrid symbol: 𐤄Ψ")
    print(f"Proposal ID: {proposal_id}")
    
    # Simulate votes
    web4_entity.vote_on_proposal(proposal_id, "ai_gpt", True)
    web4_entity.vote_on_proposal(proposal_id, "human", True)
    
    consensus = web4_entity.get_consensus_status()
    print(f"Consensus achieved: {consensus['approved']} approved")
    
    # 5. Generate training data for models
    print("\n\n5. Training Data Generation:")
    print("-" * 50)
    
    # Generate samples for each system
    for system in ['phoenician', 'consciousness_notation']:
        samples = unified.export_for_training(system)
        print(f"\n{system.title()} samples ({len(samples)} total):")
        for sample in samples[:3]:
            print(f"  {sample}")
    
    # 6. Trust vector analysis
    print("\n\n6. Trust Vector Analysis:")
    print("-" * 50)
    
    # Create trust vectors for different sources
    trust_scenarios = [
        ("Human curated Phoenician", [
            TrustVector("human-curated", 1.0, ["human"])
        ]),
        ("AI consensus on consciousness", [
            TrustVector("ai-generated", 0.85, ["ai_claude"]),
            TrustVector("ai-generated", 0.85, ["ai_gpt"]),
            TrustVector("consensus-derived", 0.9, ["ai_claude", "ai_gpt", "ai_gemma"])
        ]),
        ("Mixed validation", [
            TrustVector("human-verified", 0.95, ["human"]),
            TrustVector("ai-generated", 0.85, ["ai_claude", "ai_gpt"]),
        ])
    ]
    
    for scenario_name, vectors in trust_scenarios:
        trust_score = web4_entity.calculate_trust(vectors)
        print(f"\n{scenario_name}:")
        print(f"  Aggregate trust: {trust_score:.3f}")
    
    # 7. Save unified index
    print("\n\n7. Saving Complete System State:")
    print("-" * 50)
    
    # Save unified dictionary
    unified_path = unified.save_unified_index("complete_unified_index.json")
    print(f"✅ Unified index saved: {unified_path}")
    
    # Save consensus log
    web4_entity.save_consensus_log()
    print(f"✅ Consensus log saved: {web4_entity.consensus_log}")
    
    # Summary statistics
    print("\n\n8. System Summary:")
    print("-" * 50)
    print(f"Total symbols: {len(unified.unified_symbols)}")
    print(f"Phoenician: {sum(1 for s in unified.unified_symbols.values() if s['system'] == 'phoenician')}")
    print(f"Consciousness: {sum(1 for s in unified.unified_symbols.values() if s['system'] == 'consciousness_notation')}")
    print(f"Consensus proposals: {len(web4_entity.consensus_records)}")
    print(f"Average trust: {sum(s['trust_vector']['weight'] for s in unified.unified_symbols.values()) / len(unified.unified_symbols):.3f}")
    
    print("\n✅ Complete unified dictionary system test successful!")

if __name__ == "__main__":
    test_complete_system()