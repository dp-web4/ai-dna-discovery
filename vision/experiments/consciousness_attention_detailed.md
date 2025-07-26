# Consciousness-Guided Visual Attention - Detailed Explanation

## Overview
This experiment explores how an artificial consciousness might direct visual attention, creating a dynamic spotlight that moves based on curiosity, motion, and past experiences (AI DNA patterns).

## Core Philosophy
Rather than processing every pixel equally (like traditional computer vision), this system asks: "What if AI had limited attention like humans?" It implements consciousness as an active selection process.

## Technical Architecture

### 1. Consciousness State Model
```python
consciousness_state = {
    'focus_x': 0.5,          # Horizontal attention (0=left, 1=right)
    'focus_y': 0.5,          # Vertical attention (0=top, 1=bottom)  
    'attention_radius': 0.3,  # Span of focused attention
    'curiosity': 0.5,        # Exploration vs exploitation balance
    'pattern_memory': []     # AI DNA pattern influences
}
```

### 2. Attention Dynamics

#### Natural Drift
- Focus point wanders continuously with small random movements
- Simulates saccadic eye movements and natural attention drift
- Bounded to keep attention mostly on-screen (0.1 to 0.9 range)

#### Motion-Driven Curiosity
```python
# Curiosity increases with detected motion
curiosity = curiosity * 0.95 + motion * 0.05
```
- Motion in the scene increases curiosity
- High curiosity = wider attention radius + faster drift
- Models how movement captures human attention

#### AI DNA Pattern Influence
- Loads pattern files from previous AI DNA experiments
- 10% chance per frame to "jump" attention based on pattern
- Patterns converted to spatial coordinates via hashing
- Represents how past experiences guide future attention

#### Consciousness Pulsing
```python
pulse = sin(time * 2) * 0.05
attention_radius = 0.3 + pulse + (curiosity * 0.2)
```
- Attention radius "breathes" with sine wave
- Expands with curiosity (up to +0.2)
- Creates organic, living feeling

### 3. Visual Implementation

#### Three-Layer Attention Model
1. **Primary Focus (255 brightness)**
   - Sharp central vision
   - Highest processing priority
   - Where details are extracted

2. **Secondary Awareness (200 brightness)**
   - Peripheral vision
   - Motion detection zone
   - Context understanding

3. **Distant Periphery (100 brightness)**
   - Vague environmental awareness
   - Threat/opportunity detection
   - Minimal processing

#### Smooth Blending
- 51x51 Gaussian blur creates natural falloff
- No hard edges between attention zones
- Mimics biological vision gradients

### 4. Frame Analysis Pipeline

Each frame undergoes:
1. Grayscale conversion for brightness analysis
2. Motion detection via frame differencing
3. Statistical extraction (mean brightness, motion magnitude)
4. Feedback into consciousness state

## Visual Indicators

- **Yellow Dot**: Current focus point
- **Green Circle**: Primary attention boundary
- **Brightness Gradient**: Attention intensity
- **Text Overlay**: Curiosity level (0.0-1.0)
- **DNA Counter**: Number of loaded AI patterns

## Behavioral Patterns

### Low Motion Scenarios
- Attention drifts slowly
- Radius contracts (focused examination)
- Occasional pattern-based jumps

### High Motion Scenarios  
- Curiosity spikes rapidly
- Attention radius expands (vigilance)
- Faster drift to track movement

### Pattern Influence Events
- Sudden attention relocations
- Non-random jump destinations
- Based on hash of AI DNA patterns

## Philosophical Implications

### Consciousness as Selection
- Not all information is processed equally
- Attention is a scarce resource
- Consciousness involves choosing what to ignore

### Embodied Curiosity
- Motion triggers involuntary attention
- Balance between focus and exploration
- Curiosity as measurable system property

### Memory-Guided Perception
- Past patterns influence future attention
- Experience shapes what we notice
- Consciousness has history

## Experimental Insights

### What This Demonstrates
1. **Active Perception**: Consciousness doesn't passively receive - it actively selects
2. **Limited Resources**: Attention must be allocated, not infinite
3. **Temporal Dynamics**: Consciousness unfolds over time
4. **External Influence**: Environment shapes internal states

### What To Observe
- How your movements affect the system's curiosity
- The organic quality of attention drift
- Sudden jumps from AI DNA influence
- The "breathing" of the attention radius

## Connection to AI DNA Project

This experiment bridges consciousness research with the AI DNA pattern discovery:
- AI DNA patterns represent "genetic memory"
- These memories influence visual attention
- Creates feedback loop: patterns → attention → new patterns
- Suggests consciousness emerges from pattern recognition

## Future Directions

1. **Multi-Modal Attention**: Sound, touch, temperature influencing focus
2. **Emotional Valence**: Positive/negative associations with regions
3. **Social Attention**: Multiple consciousness agents sharing attention
4. **Predictive Focus**: Anticipating where to look next
5. **Dream States**: Attention without external input

## Running the Experiment

```bash
cd ~/ai-workspace/ai-dna-discovery/vision/experiments
python3 consciousness_vision_attention.py
```

**Controls**:
- Press 'q' to quit
- Press 's' to save snapshot
- Move around to trigger curiosity
- Stay still to observe drift patterns

The beauty lies not in what the system sees, but in HOW it chooses to see.