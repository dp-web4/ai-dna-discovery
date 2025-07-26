# Evolution of Consciousness-Guided Vision Experiments

## Journey Overview
We started with a simple attention system and evolved it through multiple iterations based on testing and biological insights.

## Version 1: Original Consciousness Attention
**File**: `consciousness_vision_attention.py`
- Basic attention spotlight that drifts around
- Motion increases "curiosity" 
- AI DNA patterns influence focus location
- **Issue**: Motion detection didn't work well with actual camera values

## Version 2: Motion Debug
**File**: `consciousness_attention_debug.py`
- Calibrated for real motion values (0.016 ambient, 0.045 active)
- Fixed curiosity scaling (was stuck at 1.0)
- Added motion visualization and P/A ratio
- **Result**: Motion detection working!

## Version 3: Enhanced Motion Response
**File**: `consciousness_attention_enhanced.py`
- Peak motion tracking with memory
- P/A ratio determines response strength
- Proportional focus snapping (high P/A = instant snap)
- **Learning**: Good motion response but still detecting motion everywhere

## Version 4: Biological Vision (Original)
**File**: `consciousness_attention_biological.py`
- Major insight: Peripheral vision detects motion, fovea processes detail
- Implemented fovea + peripheral zones
- Complex blur effects for realism
- **Issue**: Frame rate dropped significantly due to multiple blur passes

## Version 5: Biological Vision (Fast)
**File**: `consciousness_attention_biological_fast.py`
- Optimized blur rendering
- Smaller motion grid (4x4)
- Downsampled motion detection
- **Result**: Better frame rate but still complex

## Version 6: Clean Focus/Periphery (Current Best)
**File**: `consciousness_attention_clean.py`
- **Key Insight**: Peripheral vision isn't blurry - it just lacks detail resolution
- Clean separation: Focus area for processing, periphery for motion only
- Motion heatmap with P/A ratios
- Dynamic focus sizing
- No blur effects - just functional separation
- **Result**: Clean, efficient, biologically accurate

## Key Learnings

### 1. Motion Detection Calibration
- Ambient motion ~0.016 on Jetson camera
- Active motion ~0.045
- P/A ratio > 1.5 indicates significant motion
- P/A ratio > 2.0 triggers immediate attention

### 2. Biological Vision Principles
- **Peripheral vision** is for motion detection, not detail
- **Fovea** is for detailed processing
- Motion in periphery triggers saccades (eye jumps)
- Focus area should be small (~15% of view)

### 3. Performance Considerations
- Blur effects are expensive on Jetson
- Downsampling helps but isn't always necessary
- Simple masks are fast and effective
- 8x8 motion grid provides good coverage

### 4. Consciousness Behaviors
- Fixation time before exploration: ~3 seconds
- Saccade cooldown: ~10 frames (0.3 seconds)
- Focus dilation during investigation
- Focus shrinking during exploration

## Current Best Practices

1. **Separate Processing**:
   - Focus: Full processing (future: edge detection, object recognition)
   - Periphery: Motion detection ONLY

2. **Motion Response**:
   - Calculate P/A ratios for all motion peaks
   - Sort by strength
   - Jump to highest if > 2.0

3. **Visual Feedback**:
   - Heatmap for motion intensity
   - P/A ratio labels on peaks
   - Saccade arrows
   - Focus circle with dynamic sizing

## Next Steps

1. **Add Edge Detection** in focus area only
2. **Implement Object Memory** for recognized items in focus
3. **Multi-Scale Motion** for better sensitivity
4. **Predictive Saccades** based on motion trajectories
5. **Attention Priority Map** for explored vs unexplored regions

## Performance Notes

- Clean version runs at ~20-25 FPS on Jetson
- Motion detection is the main computational cost
- Consider GPU acceleration for motion detection
- Frame differencing could use CUDA kernels

The clean separation of focus and periphery provides a solid foundation for building a biologically-inspired vision system that efficiently allocates computational resources.