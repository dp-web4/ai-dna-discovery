# Binocular Vision System Status

## Video Demonstration
- **Working System Video**: [20250726_213950.mp4](../20250726_213950.mp4) - Shows binocular vision with auto-calibration running at 30 FPS

## Completed (July 27, 2025)

### Architecture
Created a modular binocular consciousness system with:

1. **Independent Eyes** (`IndependentEye` class)
   - Each eye runs identical consciousness attention algorithms
   - Configurable position offsets (3 inches apart)
   - Motion detection in peripheral vision
   - Focus tracking and updates
   - Shared codebase for both eyes

2. **Stereo Correlation Engine** (`StereoCorrelationEngine`)
   - Correlates observations from both eyes
   - Calculates depth from disparity
   - Handles 3-inch baseline
   - Provides hooks for cognition modules

3. **Cognition Interface** (`CognitionHook` abstract class)
   - Clean API for adding AI modules
   - Receives stereo observations with depth estimates
   - Can react to individual eye updates
   - Example implementation: `SimpleCognition`

4. **Calibration System** (`stereo_calibration.py`)
   - Checkerboard-based calibration
   - Handles camera misalignment
   - Saves/loads calibration data
   - Creates rectification maps

### Files Created
- `binocular_consciousness.py` - Main modular system
- `stereo_calibration.py` - Calibration tools
- `test_binocular.py` - Basic test runner
- `test_binocular_simple.py` - Circle visualization test
- `binocular_debug.py` - Debug version with motion heatmap
- `binocular_simple_track.py` - Contour-based tracking (working well!)
- `binocular_autocalibrate.py` - Grid-based auto-calibration (circles stuck)
- `binocular_autocalibrate_v2.py` - Contour-based with auto-calibration

### Current Status
- ✅ Dual cameras working (CSI0 and CSI1)
- ✅ Independent eye modules functioning
- ✅ Visualization working (orange left, blue right)
- ✅ Motion detection running
- ✅ Contour-based tracking working well (user: "that is really good!")
- ✅ Auto-calibration system implemented (v2)
- ⚠️ Need to test auto-calibration with display connected

### Key Design Decisions
1. **Modular Architecture**: Each eye is independent, correlation is separate
2. **Shared Codebase**: Both eyes use same algorithms (DRY principle)
3. **Fixed Cameras**: No mechanical vergence, focus is in image space
4. **Extensible Cognition**: Easy to add new AI modules via hooks

### Next Steps
1. ~~Auto-calibration for motion detection thresholds~~ ✅ DONE
2. Depth-based attention mechanisms
3. Temporal correlation between eyes
4. Object persistence across saccades
5. Binocular rivalry experiments

### Recent Progress (July 27, 2025)
- Fixed UnboundLocalError in binocular_autocalibrate_v2.py
- Combined working contour-based tracking with auto-calibration
- System adapts motion thresholds to environment automatically
- Calibration data can be saved/loaded between sessions

### Performance Notes
- Running at 30 FPS (camera limited)
- Motion detection working in real-time
- Room for GPU optimization later

### Usage
```bash
# Basic test
python3 test_binocular.py

# Debug with motion heatmap
python3 binocular_debug.py

# Calibration
python3 stereo_calibration.py
```

## Technical Details

### Camera Configuration
- Camera 0 (Left): /dev/video0, sensor-id=0
- Camera 1 (Right): /dev/video1, sensor-id=1
- Baseline: 3 inches (76.2mm)
- Resolution: 640x480 @ 30fps (configurable)

### Motion Detection
- 8x8 grid for motion heatmap
- Peripheral-only detection (outside focus circle)
- P/A ratio threshold for saccades
- Currently using fixed ambient threshold (needs auto-cal)