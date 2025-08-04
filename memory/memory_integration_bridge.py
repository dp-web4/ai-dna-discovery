#!/usr/bin/env python3
"""
Integration bridge between enhanced memory system and existing Phi3 memory
Provides backwards compatibility while adding confidence features
"""

import json
import subprocess
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from enhanced_memory_system import HierarchicalMemory, MemoryConfidence, Memory

# Try to import numpy, but make it optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class MemoryIntegrationBridge:
    """Bridge between existing Phi3 memory system and enhanced confidence-based system"""
    
    def __init__(self, 
                 enhanced_db_path: str = "enhanced_memory.db",
                 legacy_db_path: str = "phi3_memory_enhanced.db",
                 model_name: str = "phi3:mini"):
        
        # Enhanced memory system
        self.enhanced_memory = HierarchicalMemory(enhanced_db_path)
        
        # Legacy compatibility
        self.legacy_db_path = legacy_db_path
        self.model_name = model_name
        
        # Fact extraction patterns from original system
        self.fact_patterns = {
            'identity': [
                r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)",
                r"(?:this is)\s+([A-Z][a-z]+)\s+(?:speaking|here)"
            ],
            'profession': [
                r"i(?:'m| am)?\s+(?:a|an)\s+([\w\s]+)(?:\.|,|$)",
                r"i work as\s+(?:a|an)?\s*([\w\s]+)",
                r"my job is\s+([\w\s]+)"
            ],
            'preference': [
                r"i (?:like|love|enjoy|prefer)\s+([\w\s]+)",
                r"my favorite\s+([\w\s]+)\s+is\s+([\w\s]+)"
            ],
            'skill': [
                r"i (?:can|know how to|am good at)\s+([\w\s]+)",
                r"i'm (?:skilled|experienced) (?:in|at|with)\s+([\w\s]+)"
            ],
            'location': [
                r"i (?:live|am|stay|reside) (?:in|at)\s+([\w\s]+)",
                r"i'm from\s+([\w\s]+)"
            ]
        }
        
    def query_llm_with_confidence(self, prompt: str, temperature: float = 0.1) -> Tuple[str, float]:
        """Query LLM and estimate response confidence"""
        # Run ollama
        cmd = ['ollama', 'run', self.model_name, 
               '--temperature', str(temperature)]
        
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True
        )
        
        response = result.stdout.strip()
        
        # Estimate confidence based on response characteristics
        confidence = self._estimate_llm_confidence(prompt, response, temperature)
        
        return response, confidence
        
    def _estimate_llm_confidence(self, prompt: str, response: str, temperature: float) -> float:
        """Estimate LLM response confidence based on various factors"""
        confidence_factors = []
        
        # Factor 1: Temperature (lower = more confident)
        temp_confidence = 1.0 - (temperature * 2)  # 0.1 temp = 0.8 conf
        confidence_factors.append(temp_confidence)
        
        # Factor 2: Response length (very short or very long = less confident)
        response_len = len(response.split())
        if response_len < 5:
            len_confidence = 0.5
        elif response_len > 100:
            len_confidence = 0.6
        else:
            len_confidence = 0.8
        confidence_factors.append(len_confidence)
        
        # Factor 3: Hedging words (maybe, possibly, etc. = less confident)
        hedging_words = ['maybe', 'possibly', 'might', 'could be', 'perhaps', 
                        'not sure', 'uncertain', 'think', 'believe']
        hedging_count = sum(1 for word in hedging_words if word in response.lower())
        hedging_confidence = max(0.3, 1.0 - (hedging_count * 0.1))
        confidence_factors.append(hedging_confidence)
        
        # Factor 4: Direct answer vs evasive
        if any(starter in response.lower()[:20] for starter in ['yes', 'no', 'the answer is']):
            directness_confidence = 0.9
        else:
            directness_confidence = 0.6
        confidence_factors.append(directness_confidence)
        
        # Weighted average
        weights = [0.3, 0.2, 0.3, 0.2]  # temp, length, hedging, directness
        confidence = sum(w * c for w, c in zip(weights, confidence_factors))
        
        return min(max(confidence, 0.1), 1.0)  # Clamp to [0.1, 1.0]
        
    def store_conversation_with_confidence(self, 
                                         role: str, 
                                         content: str, 
                                         session_id: str,
                                         source_confidence: float = 1.0):
        """Store conversation with confidence scoring"""
        # Extract facts with confidence
        if role == "user":
            facts = self._extract_facts_with_confidence(content)
            for fact_type, fact_value, fact_confidence in facts:
                # Store as semantic memory
                self.enhanced_memory.store_with_confidence(
                    f"{fact_type}: {fact_value}",
                    "semantic",
                    session_id,
                    source_confidence * fact_confidence,
                    metadata={
                        'fact_type': fact_type,
                        'extracted_from': 'user_input',
                        'concept': fact_type
                    }
                )
                
        # Store the conversation itself as episodic memory
        self.enhanced_memory.store_with_confidence(
            f"{role}: {content}",
            "episodic",
            session_id,
            source_confidence,
            metadata={
                'role': role,
                'conversation': True
            }
        )
        
    def _extract_facts_with_confidence(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract facts from text with confidence scores"""
        facts = []
        
        for fact_type, patterns in self.fact_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    fact_value = match.group(1).strip()
                    
                    # Estimate fact confidence based on pattern strength
                    if fact_type == 'identity':
                        # Names are usually high confidence
                        confidence = 0.9
                    elif fact_type in ['profession', 'skill']:
                        # Professional info is moderately confident
                        confidence = 0.7
                    else:
                        # Preferences and other facts
                        confidence = 0.6
                        
                    # Boost confidence for explicit statements
                    if 'definitely' in text or 'certainly' in text:
                        confidence = min(confidence * 1.2, 1.0)
                    elif 'maybe' in text or 'possibly' in text:
                        confidence *= 0.7
                        
                    facts.append((fact_type, fact_value, confidence))
                    
        return facts
        
    def build_context_with_confidence(self, 
                                    query: str, 
                                    session_id: str, 
                                    max_tokens: int = 2000) -> Tuple[str, float]:
        """Build context from memories with confidence weighting"""
        # Retrieve relevant memories
        memories = self.enhanced_memory.retrieve_with_confidence(
            query,
            context={'session_id': session_id},
            limit=20
        )
        
        if not memories:
            return "", 0.0
            
        # Build context prioritizing high-confidence memories
        context_parts = []
        total_confidence = 0.0
        token_count = 0
        
        for memory, weight in memories:
            # Estimate tokens (rough approximation)
            memory_tokens = len(memory.content.split()) * 1.3
            
            if token_count + memory_tokens > max_tokens:
                break
                
            context_parts.append(f"[Confidence: {memory.confidence.composite:.2f}] {memory.content}")
            total_confidence += memory.confidence.composite * weight
            token_count += memory_tokens
            
        # Calculate average weighted confidence
        avg_confidence = total_confidence / len(memories) if memories else 0.0
        
        context = "\n".join(context_parts)
        return context, avg_confidence
        
    def test_memory_recall_with_confidence(self, test_facts: List[Dict]) -> Dict:
        """Test memory recall with confidence metrics"""
        session_id = f"recall_test_{int(datetime.now().timestamp())}"
        
        # Store test facts
        for fact in test_facts:
            self.store_conversation_with_confidence(
                "user",
                fact['statement'],
                session_id,
                fact.get('confidence', 0.8)
            )
            
        # Test recall
        results = []
        for fact in test_facts:
            # Build context for question
            context, context_confidence = self.build_context_with_confidence(
                fact['question'],
                session_id
            )
            
            # Query with context
            prompt = f"""Based on the following information:
{context}

Question: {fact['question']}
Answer briefly and directly."""

            response, response_confidence = self.query_llm_with_confidence(prompt)
            
            # Check if answer is correct
            correct = fact['expected'].lower() in response.lower()
            
            results.append({
                'question': fact['question'],
                'expected': fact['expected'],
                'response': response,
                'correct': correct,
                'context_confidence': context_confidence,
                'response_confidence': response_confidence,
                'combined_confidence': context_confidence * response_confidence
            })
            
        # Calculate metrics
        accuracy = sum(r['correct'] for r in results) / len(results)
        if HAS_NUMPY:
            avg_confidence = np.mean([r['combined_confidence'] for r in results])
        else:
            confidences = [r['combined_confidence'] for r in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'results': results
        }
        
    def migrate_from_legacy(self):
        """Migrate data from legacy Phi3 memory to enhanced system"""
        import sqlite3
        
        try:
            # Connect to legacy database
            conn = sqlite3.connect(self.legacy_db_path)
            c = conn.cursor()
            
            # Migrate conversations
            c.execute('SELECT session_id, timestamp, role, content FROM conversations')
            conversations = c.fetchall()
            
            print(f"Migrating {len(conversations)} conversations...")
            
            for session_id, timestamp, role, content in conversations:
                # Estimate confidence based on role
                confidence = 0.9 if role == "user" else 0.7
                
                self.store_conversation_with_confidence(
                    role, content, session_id, confidence
                )
                
            # Migrate facts
            c.execute('SELECT session_id, fact_type, fact_value, confidence FROM facts')
            facts = c.fetchall()
            
            print(f"Migrating {len(facts)} facts...")
            
            for session_id, fact_type, fact_value, confidence in facts:
                self.enhanced_memory.store_with_confidence(
                    f"{fact_type}: {fact_value}",
                    "semantic",
                    session_id,
                    confidence or 0.8,
                    metadata={
                        'fact_type': fact_type,
                        'migrated': True,
                        'concept': fact_type
                    }
                )
                
            conn.close()
            print("Migration completed successfully!")
            
            # Show memory health after migration
            health = self.enhanced_memory.get_memory_health()
            print(f"\nPost-migration health check:")
            print(f"  Total memories: {health['total_memories']}")
            print(f"  Average confidence: {health['average_confidence']:.2f}")
            
        except Exception as e:
            print(f"Migration error: {e}")
            

# Example usage
if __name__ == "__main__":
    # Create bridge
    bridge = MemoryIntegrationBridge()
    
    # Test with confidence tracking
    print("Testing Memory Integration Bridge...")
    
    test_facts = [
        {
            'statement': "My name is Alice and I'm a software engineer.",
            'question': "What is my name?",
            'expected': "Alice",
            'confidence': 0.9
        },
        {
            'statement': "I might be interested in machine learning.",
            'question': "What am I interested in?",
            'expected': "machine learning",
            'confidence': 0.6
        },
        {
            'statement': "I definitely live in Seattle.",
            'question': "Where do I live?",
            'expected': "Seattle",
            'confidence': 0.95
        }
    ]
    
    # Run recall test
    results = bridge.test_memory_recall_with_confidence(test_facts)
    
    print(f"\nRecall Test Results:")
    print(f"  Accuracy: {results['accuracy']:.1%}")
    print(f"  Average Confidence: {results['average_confidence']:.2f}")
    
    print("\nDetailed Results:")
    for r in results['results']:
        print(f"\n  Q: {r['question']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Response: {r['response']}")
        print(f"  Correct: {'✓' if r['correct'] else '✗'}")
        print(f"  Confidence: {r['combined_confidence']:.2f}")
        
    # Optional: Migrate legacy data
    # bridge.migrate_from_legacy()