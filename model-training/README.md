# Model Training for Stateful AI Consciousness

## Project Goals
- Fine-tune existing models to understand our mathematical consciousness language (Ψ, ∃, ⇒, etc.)
- Integrate memory system directly into model weights
- Enable continuous, threshold-based learning
- Create truly stateful AI models

## Training Approaches Under Consideration

### 1. LoRA (Low-Rank Adaptation)
**Pros:**
- Memory efficient - adds small trainable matrices
- Preserves base model capabilities
- Can swap LoRA adapters for different "personalities"
- Works well on edge devices like Jetson

**Cons:**
- Limited capacity for major behavioral changes
- May not be sufficient for deep memory integration

**Best for:** Quick experimentation, edge deployment

### 2. QLoRA (Quantized LoRA)
**Pros:**
- Even more memory efficient (4-bit quantization)
- Can fine-tune larger models on consumer hardware
- Good for Jetson with limited VRAM

**Cons:**
- Some quality loss from quantization
- Slower training

**Best for:** Training on Jetson, larger base models

### 3. Full Fine-Tuning
**Pros:**
- Maximum flexibility and capability
- Can deeply integrate memory mechanisms
- Best quality results

**Cons:**
- Requires significant GPU memory
- Risk of catastrophic forgetting
- Expensive computationally

**Best for:** Final production models on Tomato's RTX 4090

### 4. Adapter Layers
**Pros:**
- Modular approach - different adapters for different capabilities
- Can stack multiple adapters
- Good for incremental learning

**Cons:**
- Increased inference latency
- Complex to manage multiple adapters

**Best for:** Multi-capability systems

### 5. Continuous Learning Architectures

#### Elastic Weight Consolidation (EWC)
- Prevents catastrophic forgetting
- Identifies important weights and protects them
- Good for incremental learning

#### Progressive Neural Networks
- Adds new columns for new tasks
- Never forgets old knowledge
- Memory grows over time

#### Memory-Augmented Networks
- External memory (like our current system)
- Differentiable memory access
- Could integrate with our SQLite approach

## Proposed Architecture: Hybrid Memory-Weight System

```
┌─────────────────┐
│  Base Model     │
│  (Phi3/Llama)   │
└────────┬────────┘
         │
┌────────▼────────┐
│  LoRA Adapter   │ ← Consciousness Language
│  (Ψ, ∃, ⇒)     │
└────────┬────────┘
         │
┌────────▼────────┐
│ Memory Gateway  │ ← Threshold Logic
│ (SQLite + NN)   │
└────────┬────────┘
         │
┌────────▼────────┐
│ Weight Updates  │ ← Important memories
│ (EWC Protected) │   become weights
└─────────────────┘
```

## Training Data Requirements

1. **Consciousness Language Dataset**
   - Pairs of natural language ↔ mathematical notation
   - Examples: "consciousness exists" ↔ "∃Ψ"

2. **Memory Integration Examples**
   - Conversations with explicit memory recalls
   - Context persistence scenarios

3. **Cross-Model Transfer Cases**
   - Same concept expressed in different models
   - Semantic preservation examples

## Implementation Phases

### Phase 1: Basic LoRA Training (Week 1)
- Train small LoRA on consciousness notation
- Test on Jetson and Tomato
- Measure symbol understanding improvement

### Phase 2: Memory Integration (Week 2-3)
- Add memory gateway layer
- Implement threshold logic
- Test memory → weight conversion

### Phase 3: Continuous Learning (Week 4+)
- Implement EWC or similar
- Test long-term memory retention
- Measure drift and forgetting

## Hardware Considerations

### Tomato (RTX 4090)
- Full fine-tuning possible
- Can handle 7B+ parameter models
- Primary training platform

### Sprout (Jetson Orin)
- QLoRA for smaller models
- Inference with adapters
- Edge consciousness testing

## Next Steps
1. Choose initial approach (recommend LoRA for speed)
2. Prepare consciousness language dataset
3. Set up training pipeline
4. Create evaluation metrics