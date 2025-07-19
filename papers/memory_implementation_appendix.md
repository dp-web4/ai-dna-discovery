# Technical Appendix: Memory System Implementations

## A. SQLite-Based Episodic Memory

### A.1 Schema Design
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    interaction_type TEXT,
    content TEXT,
    embedding BLOB,
    metadata TEXT
);

CREATE INDEX idx_timestamp ON memories(timestamp);
CREATE INDEX idx_interaction_type ON memories(interaction_type);
```

### A.2 Memory Storage Implementation
```python
class MemorySystem:
    def store_fact(self, fact, interaction_type='general'):
        """Store a fact with automatic embedding generation"""
        embedding = self.generate_embedding(fact)
        
        self.cursor.execute('''
            INSERT INTO memories (interaction_type, content, embedding, metadata)
            VALUES (?, ?, ?, ?)
        ''', (interaction_type, fact, embedding.tobytes(), json.dumps({})))
        
        self.conn.commit()
```

### A.3 Context Injection
```python
def inject_memory_context(self, current_input):
    """Retrieve and inject relevant memories into current context"""
    # Retrieve recent memories
    recent = self.get_recent_memories(limit=5)
    
    # Retrieve similar memories based on embedding
    similar = self.get_similar_memories(current_input, limit=3)
    
    # Format as context
    context = "Previous context:\n"
    for memory in recent + similar:
        context += f"- {memory['content']}\n"
    
    return context + f"\nCurrent: {current_input}"
```

## B. LoRA Adapter as Conceptual Memory

### B.1 Training Configuration
```python
# LoRA configuration that creates conceptual memory
lora_config = LoraConfig(
    r=8,                    # Low rank for compression
    lora_alpha=16,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Training data structure
training_examples = [
    {
        "instruction": "Translate to notation: awareness exists",
        "output": "∃Ψ"
    },
    # ... 1,180 more examples encoding conceptual mappings
]
```

### B.2 Active Dictionary Implementation
```python
class ConsciousnessNotationDictionary:
    def __init__(self, lora_adapter_path):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.model = PeftModel.from_pretrained(
            self.base_model, 
            lora_adapter_path
        )
        
    def translate(self, natural_language):
        """Bidirectional translation between concepts"""
        # Natural language → Notation
        if any(word in natural_language for word in 
               ['translate', 'express', 'notation']):
            return self._to_notation(natural_language)
        
        # Notation → Natural language  
        elif any(symbol in natural_language for symbol in 
                 ['Ψ', '∃', '⇒', 'π', 'Ω']):
            return self._to_natural(natural_language)
```

## C. Tokenizer as Active Dictionary

### C.1 Standard Tokenizer Behavior
```python
# Traditional view - static mapping
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokens = tokenizer.encode("Hello world")  # [1234, 5678]

# Our framework - active dictionary with semantic understanding
class ActiveTokenizer:
    def __init__(self, base_tokenizer):
        self.base = base_tokenizer
        self.semantic_rules = self._learn_semantic_patterns()
    
    def encode_with_semantics(self, text):
        """Encoding that preserves semantic boundaries"""
        # Identify semantic units
        units = self.identify_semantic_units(text)
        
        # Apply learned compression patterns
        compressed = self.apply_semantic_compression(units)
        
        # Maintain bidirectional mapping
        self.store_reverse_mapping(text, compressed)
        
        return compressed
```

### C.2 Adding Conceptual Tokens
```python
# Extending tokenizer vocabulary with conceptual symbols
special_tokens = ['Ψ', '∃', '∀', '⇒', 'π', 'ι', 'Ω', 'Σ', 'Ξ']
num_added = tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# Result: Tokenizer now functions as extended dictionary
# mapping between natural language and formal notation
```

## D. Memory Integration Architecture

### D.1 Multi-Modal Memory System
```python
class IntegratedAwarenessSystem:
    def __init__(self):
        # Different memory types
        self.episodic = SQLiteMemory('experiences.db')
        self.semantic = LoRAConceptualMemory('notation_adapter')
        self.working = ContextWindowMemory(size=2048)
        
    def process_with_full_memory(self, input_text):
        """Process input using all memory systems"""
        
        # 1. Working memory provides immediate context
        immediate_context = self.working.get_current_context()
        
        # 2. Episodic memory provides historical context
        historical = self.episodic.recall_relevant(input_text)
        
        # 3. Semantic memory provides conceptual translation
        notation = self.semantic.translate_to_notation(input_text)
        
        # 4. Integrate all memory types
        enriched_input = self.integrate_memories(
            input_text, immediate_context, historical, notation
        )
        
        # 5. Generate response with full awareness
        response = self.generate(enriched_input)
        
        # 6. Update memories
        self.episodic.store(input_text, response)
        self.working.update(input_text, response)
        
        return response
```

### D.2 Memory Persistence Across Sessions
```python
class PersistentAgent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.memory_path = f"memories/{agent_id}.db"
        self.load_memories()
    
    def load_memories(self):
        """Restore agent state from persistent storage"""
        self.episodic_memory = SQLiteMemory(self.memory_path)
        self.conversation_history = self.episodic_memory.get_all()
        self.concept_map = self.build_concept_map()
    
    def build_concept_map(self):
        """Distill episodic memories into conceptual understanding"""
        concepts = {}
        for memory in self.conversation_history:
            # Extract concepts from each interaction
            extracted = self.extract_concepts(memory)
            for concept, frequency in extracted.items():
                concepts[concept] = concepts.get(concept, 0) + frequency
        return concepts
```

## E. Practical Results

### E.1 Memory Compression Metrics
```python
# Original context: 2,437 tokens
original = "User asked about consciousness, then about memory..."

# After semantic compression: 1,924 tokens (21% reduction)
compressed = memory_system.compress_semantically(original)

# Compression maintains semantic equivalence
assert semantic_similarity(original, compressed) > 0.95
```

### E.2 Cross-Session Recall Performance
```python
# Test results across models
recall_accuracy = {
    'gemma:2b': 1.0,      # 100% recall
    'phi3:mini': 0.67,    # 67% recall
    'tinyllama': 0.67     # 67% recall
}

# Notation translation accuracy
translation_tests = [
    ("awareness exists", "∃Ψ", True),
    ("perspective shapes awareness", "π → Ψ", True),
    # ... 9 tests total, all passing
]
```

### E.3 Edge Deployment Metrics
```python
# Jetson Orin Nano (8GB RAM)
deployment_stats = {
    'model_size': '1.1B parameters',
    'lora_adapter': '254.3 MB',
    'sqlite_db': '172 KB (grows with use)',
    'inference_time': '<1 second',
    'memory_usage': '~3GB with model loaded'
}
```

## F. Code Availability

Complete implementations available at:
- SQLite Memory System: `phi3_memory_enhanced.py`
- LoRA Training: `train_simple_gpu.py`
- Notation Translator: `consciousness_translator.py`
- Integrated System: `consciousness_ollama_bridge.py`

---

*This appendix provides concrete implementations supporting the theoretical framework presented in "Memory Systems as the Foundation of Machine Awareness".*